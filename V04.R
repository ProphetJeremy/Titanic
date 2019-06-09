###########################################################
# Competición Titanic Kaggle
# Algoritmo tipo: BAGGING
# Resultado: 0.76555
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
library(ROCR)
library(ranger)

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
bagggrid <-  expand.grid(mtry = c(ncol(datTrain_sca)),
                         splitrule = c("gini", "extratrees"),
                         min.node.size = seq(1,22,3))
bagg<- train(frml_All, data=datTrain_sca, metric="ROC",
             method="ranger",
             trControl=control, tuneGrid=bagggrid,
             replace = TRUE, num.trees = 300, sample.fraction = 0.75)

datos<- as.data.frame(myMetric(bagg$results))
saveRDS(datos, "./tab/bagg_tune")

png("./img/bagg_tune.png", width = 1000)
ggplot(datos, aes(min.node.size, metric, fill=ROCSD)) + geom_point(shape=21, size=2) +
    facet_grid(cols = vars(splitrule)) +
    ggtitle("Mi métrica para cada configuración") +
    scale_y_continuous(labels = function(x){format(round(x, 2), nsmall=2)})
dev.off()

# 3.3 Validación cruzada ----
bagggrid <-  expand.grid(splitrule=datos$splitrule[which(datos$metric == max(datos$metric))],
                         min.node.size=datos$min.node.size[which(datos$metric == max(datos$metric))],
                         mtry=ncol(datTrain_sca))
modelos.nombres <- c("bagg_full", "bagg_Cra", "bagg_AIC", "bagg_BIC", "bagg_Tre")
modelos.formula<- c(frml_All, frml_Vcr, frml_AIC, frml_BIC, frml_Tre)
total<- c()
for (i in 1:length(modelos.nombres)){
    set.seed(1712)
    bagggrid$mtry<-length(unlist(strsplit(as.character(modelos.formula[[i]])[3], "\\+")))
    vcr<-train(modelos.formula[[i]], data = datTrain_sca,
               method="ranger",
               replace = TRUE, num.trees = 300, sample.fraction = 0.75,
               tuneGrid=bagggrid, metric="ROC",
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
saveRDS(comparativa, "./tab/bagg_compared")

# 3.4 Boxplot ----
cv_all<- readRDS("./tab/cv_all")
cv_all<- rbind(cv_all, total)
ggplot(cv_all, aes(x=modelo, y=metric, fill=family)) + 
    geom_boxplot(outlier.colour="red", outlier.shape=20,
                 outlier.size=2) +
    stat_summary(fun.y=mean, geom="point", colour="white",fill="grey", shape=21, size=3) +
    scale_x_discrete(limits=as.vector(unique(cv_all$modelo)[order(aggregate(metric~modelo, data=cv_all, mean)$metric)])) + 
    theme_minimal() + theme_update(plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
                                   axis.text.x = element_text(size = 10, face = "bold", angle = 45)) +
    ggtitle("Mi métrica")
saveRDS(cv_all, "./tab/cv_all")

# 4. Make a Submission with the best model ----
mod_best<- readRDS("./mod/bagg_full")
test_pred <- as.factor(ifelse(predict(mod_best, datTest_sca) == "X0",0,1))
submission$Survived = test_pred
fwrite(submission, "bagg_sub.csv", row.names = FALSE)
