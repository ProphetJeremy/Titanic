###########################################################
# Competición Titanic Kaggle
# Algoritmo tipo: SVM
# Resultado: 0.78947
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
svmgrid_lin<-expand.grid(C=c(0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10))
svmgrid_pol<-expand.grid(C=c(0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10),
                         degree=c(2,3,4),
                         scale=c(0.1,0.5,1,2,5))
svmgrid_rad<-expand.grid(C=c(0.01,0.02,0.05,0.1,0.2,0.5,1,2,5,10),
                         sigma=c(0.01,0.05,0.1,0.2,0.5,1,2,5,10,30))
svm_grids<- list(svmgrid_lin, svmgrid_pol, svmgrid_rad)
svm_methods<- c("svmLinear", "svmPoly", "svmRadial")
datos<-c()
for(i in 1:length(svm_methods)){
    svm<- train(frml_All, data=datTrain_sca, metric="ROC",
                method=svm_methods[i],
                trControl=control, tuneGrid=svm_grids[[i]],
                verbose=FALSE)
    info<- as.data.frame(myMetric(svm$resample))
    info$method<- svm_methods[i]
    for(addC in c("degree", "scale", "sigma")){
        if(!(addC %in% names(info))){
            info[[addC]]<- 0
        }
    }
    datos<-rbind(datos,info)  
}
datos<- datos[order(datos$metric, na.last=FALSE),]
saveRDS(datos, "./tab/svm_tune")

p1<- ggplot(datos[which(datos$method == "svmLinear"),], aes(C, metric)) +
    geom_point(shape=21, size=2, colour="black", fill="#3d6497") +
    ggtitle("svmLinear") +
    scale_y_continuous(labels = function(x){format(round(x, 2), nsmall=2)})
p2<- ggplot(datos[which(datos$method == "svmPoly"),], aes(C, metric, fill=scale)) +
    geom_point(shape=21, size=2) +
    facet_grid(cols = vars(degree)) +
    ggtitle("svmPoly") +
    scale_y_continuous(labels = function(x){format(round(x, 2), nsmall=2)})
p3<- ggplot(datos[which(datos$method == "svmRadial"),], aes(C, metric, fill=sigma)) +
    geom_point(shape=21, size=2) +
    ggtitle("svmRadial") +
    scale_y_continuous(labels = function(x){format(round(x, 2), nsmall=2)})
gridExtra::grid.arrange(p1, p2, p3, widths = c(1, 2, 2))

## 3.21 Tuneo del mejor modelo
svmgrid_rad<-expand.grid(C=seq(0.01,1,0.08),
                         sigma=seq(0.01,2.01,0.1))
svm<- train(frml_All, data=datTrain_sca, metric="ROC",
            method="svmRadial",
            trControl=control, tuneGrid=svmgrid_rad,
            verbose=FALSE)
datos<- as.data.frame(myMetric(svm$resample))
saveRDS(datos, "./tab/svm_tune2")

ggplot(datos, aes(sigma, metric, fill=Resample)) +
    geom_point(shape=21, size=2, colour="black") +
    facet_grid(cols = vars(C)) +
    ggtitle("svmRadial") +
    scale_y_continuous(labels = function(x){format(round(x, 2), nsmall=2)})


# 3.3 Validación cruzada ----
svmgrid_rad<-expand.grid(C=0.4, sigma=0.2)
modelos.nombres <- c("svm_full", "svm_Cra", "svm_AIC", "svm_BIC", "svm_Tre")
modelos.formula<- c(frml_All, frml_Vcr, frml_AIC, frml_BIC, frml_Tre)
total<- c()
for (i in 1:length(modelos.nombres)){
    set.seed(1712)
    vcr<-train(modelos.formula[[i]], data = datTrain_sca,
               method="svmRadial",
               tuneGrid=svmgrid_rad, metric="ROC",
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
saveRDS(comparativa, "./tab/svm_compared")

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
mod_best<- readRDS("./mod/svm_AIC")
test_pred <- as.factor(ifelse(predict(mod_best, datTest_sca) == "X0",0,1))
submission$Survived = test_pred
fwrite(submission, "svm_sub.csv", row.names = FALSE)
