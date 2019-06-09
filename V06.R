###########################################################
# Competición Titanic Kaggle
# Algoritmo tipo: GBM
# Resultado: 0.76076
# Posición: No mejora
# Percentil: No mejora
# Fecha: 15 - 05 - 2019
###########################################################

setwd("~/Desktop/Santi/master_BD/machine_learning/entrega/")
setwd("/cloud/project/entrega/")
source("funcionesML.R")

# 0. Cargo paquetes ----
library(dplyr)
library(ggplot2)
library(data.table)
library(stringr)
library(tidyr)
library(gridExtra)
library(questionr)
library(caret)
library(purrr)
`%!in%` <- compose(`!`, `%in%`)

# 1. Carga de datos ----
datTrain_sca<- readRDS("train_clean_sca")
datTest_sca<- readRDS("test_clean_sca")

# 3. Modelado ----
submission<- data.frame(PassengerId = as.numeric(row.names(datTest_sca)))

## 3.1 Selección de variables ----
y_var<- "Survived"
sel_All<- c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", 
            "Embarked", "CabLetter", "CabNumber_bin", "TickNumber", "Title", 
            "SurnFreq", "FamilySize", "TickLetters_g")
sel_VCr<- c("Sex", "Fare", "TickNumber", "Title")
sel_AIC<- c("Pclass", "Age", "CabNumber_bin", "Title", "FamilySize", "TickLetters_g")
sel_BIC<- c("Pclass", "Title")
sel_Tre<- c("Title", "Sex", "FamilySize", "Parch", "SibSp", "Pclass", "Age", "Fare")

# Fórmulas
frml_All<- as.formula(paste(y_var, paste(sel_All, collapse="+"), sep="~"))
frml_Vcr<- as.formula(paste(y_var, paste(sel_VCr, collapse="+"), sep="~"))
frml_AIC<- as.formula(paste(y_var, paste(sel_AIC, collapse="+"), sep="~"))
frml_BIC<- as.formula(paste(y_var, paste(sel_BIC, collapse="+"), sep="~"))
frml_Tre<- as.formula(paste(y_var, paste(sel_Tre, collapse="+"), sep="~"))

# 3.2 Tuneo hiperparámetros ----
datTrain_sca$Survived<-make.names(datTrain_sca$Survived)
control<-trainControl(method = "LGOCV", p=0.8, number=5, savePredictions = "all",
                      summaryFunction=twoClassSummary, classProbs=TRUE, returnResamp="all",
                      verboseIter=FALSE)
gbmgrid<-expand.grid(shrinkage=seq(0.005, 0.065, 0.015),
                     n.minobsinnode=seq(1,10,3),
                     n.trees=c(500, 750, 1000, 1250),
                     interaction.depth=seq(2,3))
gbm<- train(frml_All, data=datTrain_sca, metric="ROC",
            method="gbm",
            trControl=control, tuneGrid=gbmgrid,
            distribution="bernoulli", bag.fraction=1,
            verbose=FALSE)
plot(gbm)
datos<- as.data.frame(myMetric(gbm$results))
saveRDS(datos, "./tab/gbm_tune2")

# 3.3 Validación cruzada ----
gbmgrid <-  expand.grid(shrinkage=datos$shrinkage[which(datos$metric == max(datos$metric))],
                        n.minobsinnode=datos$n.minobsinnode[which(datos$metric == max(datos$metric))],
                        n.trees=datos$n.trees[which(datos$metric == max(datos$metric))],
                        interaction.depth=datos$interaction.depth[which(datos$metric == max(datos$metric))])
modelos.nombres <- c("gbm_full", "gbm_Cra", "gbm_AIC", "gbm_BIC", "gbm_Tre")
modelos.formula<- c(frml_All, frml_Vcr, frml_AIC, frml_BIC, frml_Tre)
total<- c()
for (i in 1:length(modelos.nombres)){
    set.seed(1712)
    vcr<-train(modelos.formula[[i]], data = datTrain_sca,
               method="gbm",
               distribution="bernoulli", bag.fraction=1,
               tuneGrid=gbmgrid, metric="ROC",
               trControl = trainControl(method = "repeatedcv",
                                        number=8, repeats=10,
                                        savePredictions = "all",
                                        summaryFunction=twoClassSummary,
                                        classProbs=TRUE, returnResamp="all",
                                        verboseIter=FALSE)
    )
    mod_info<- as.data.frame(myMetric(vcr$resample))
    fam<- sub("^([A-Za-z]+)_.*", "\\1", modelos.nombres[i], perl=TRUE)
    to_total<- data.frame(metric = mod_info$metric,
                          modelo = modelos.nombres[i],
                          family = fam)
    total<-rbind(total,to_total)
    saveRDS(vcr, paste("./mod/", modelos.nombres[i], sep=""))
}

comparativa<- data.frame(row.names = unique(total$modelo))
comparativa$mean_metric<- aggregate(metric~modelo, data = total, mean)$metric
comparativa$std_metric<- aggregate(metric~modelo, data = total, sd)$metric
comparativa<- comparativa[order(comparativa$mean_metric),]
saveRDS(comparativa, "./tab/gbm_compared")

# 3.4 Boxplot ----
cv_all<- readRDS("./tab/cv_all")
cv_all<- rbind(cv_all, total)
medias<- as.data.frame(aggregate(metric~modelo, data=cv_all, mean))
ggplot(cv_all, aes(x=modelo, y=metric, fill=family)) + 
    geom_boxplot(outlier.colour="red", outlier.shape=20,
                 outlier.size=2) +
    stat_summary(fun.y=mean, geom="point", colour="white",fill="grey", shape=21, size=3) +
    scale_x_discrete(limits=as.vector(medias[order(medias$metric), c("modelo")])) + 
    theme_minimal() + theme_update(plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
                                   axis.text.x = element_text(size = 10, face = "bold", angle = 45)) +
    ggtitle("Mi métrica")
saveRDS(cv_all, "./tab/cv_all")

# 4. Make a Submission with the best model ----
mod_best<- readRDS("./mod/gbm_full")
test_pred <- as.factor(ifelse(predict(mod_best, datTest_sca) == "X0",0,1))
submission$Survived = test_pred
fwrite(submission, "gbm_sub.csv", row.names = FALSE)

# 5. Repeat process for ROC metric ----
gbmgrid <-  expand.grid(shrinkage=datos$shrinkage[which(datos$ROC == max(datos$ROC))],
                        n.minobsinnode=datos$n.minobsinnode[which(datos$ROC == max(datos$ROC))],
                        n.trees=datos$n.trees[which(datos$ROC == max(datos$ROC))],
                        interaction.depth=datos$interaction.depth[which(datos$ROC == max(datos$ROC))])
modelos.nombres <- c("gbm_full", "gbm_Cra", "gbm_AIC", "gbm_BIC", "gbm_Tre")
modelos.formula<- c(frml_All, frml_Vcr, frml_AIC, frml_BIC, frml_Tre)
total<- c()
for (i in 1:length(modelos.nombres)){
    set.seed(1712)
    vcr<-train(modelos.formula[[i]], data = datTrain_sca,
               method="gbm",
               distribution="bernoulli", bag.fraction=1,
               tuneGrid=gbmgrid, metric="ROC",
               trControl = trainControl(method = "repeatedcv",
                                        number=8, repeats=10,
                                        savePredictions = "all",
                                        summaryFunction=twoClassSummary,
                                        classProbs=TRUE, returnResamp="all",
                                        verboseIter=FALSE)
    )
    mod_info<- as.data.frame(myMetric(vcr$resample))
    fam<- sub("^([A-Za-z]+)_.*", "\\1", modelos.nombres[i], perl=TRUE)
    to_total<- data.frame(metric = mod_info$metric,
                          modelo = modelos.nombres[i],
                          family = fam)
    total<-rbind(total,to_total)
}

## 5.1 Boxplot ----
cv_all<- readRDS("./tab/rforest_CV")[,c("modelo", "metric", "family")]
cv_all<- cv_all[which(cv_all$family %!in% c("gbm")),]
cv_all<- rbind(cv_all, total)
medias<- as.data.frame(aggregate(metric~modelo, data=cv_all, mean))
ggplot(cv_all, aes(x=modelo, y=metric, fill=family)) + 
    geom_boxplot(outlier.colour="red", outlier.shape=20,
                 outlier.size=2) +
    stat_summary(fun.y=mean, geom="point", colour="white",fill="grey", shape=21, size=3) +
    scale_x_discrete(limits=as.vector(medias[order(medias$metric), c("modelo")])) + 
    theme_minimal() + theme_update(plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
                                   axis.text.x = element_text(size = 10, face = "bold", angle = 45)) +
    ggtitle("Mi métrica")

## 5.2. Make a Submission with the best model ----
mod_best<- readRDS("./mod/gbm_full")
test_pred <- as.factor(ifelse(predict(mod_best, datTest_sca) == "X0",0,1))
submission$Survived = test_pred
fwrite(submission, "gbm_sub2.csv", row.names = FALSE)
