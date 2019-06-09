###########################################################
# Competición Titanic Kaggle
# Algoritmo tipo: RANDOM FOREST
# Resultado: 0.78468
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

# Fórmulas
frml_All<- as.formula(paste(y_var, paste(sel_All, collapse="+"), sep="~"))

# 3.2 Validación cruzada ----
datTrain_sca$Survived<-make.names(datTrain_sca$Survived)
control<- trainControl(method = "repeatedcv",
                       number=8, repeats=10,
                       savePredictions = "all",
                       summaryFunction=twoClassSummary,
                       classProbs=TRUE, returnResamp="all",
                       verboseIter=FALSE)

rforestgrid <-  expand.grid(mtry = c(1:ncol(datTrain_sca)),
                            splitrule = c("gini", "extratrees"),
                            min.node.size = seq(1,15,2))

rforest<- train(frml_All, data=datTrain_sca, metric="ROC",
                method="ranger",
                trControl=control, tuneGrid=rforestgrid,
                replace = TRUE, num.trees = 500, sample.fraction = 0.8)
saveRDS(rforest, "./mod/rforest")

## 3.3 Desempeño según parámetros ----
datos<- as.data.frame(myMetric(rforest$results))
datos[nrow(datos),]
saveRDS(datos, "./tab/rforest_tune")
png("./img/rforest_tune.png", width = 1000)
ggplot(datos, aes(mtry, metric, fill=ROCSD)) + geom_point(shape=21, size=2) +
    facet_grid(rows = vars(splitrule), cols = vars(min.node.size)) +
    ggtitle("Mi métrica para cada configuración") +
    scale_y_continuous(labels = function(x){format(round(x, 2), nsmall=2)})
dev.off()

## 3.4 Desempeño del modelo ----
mod_info<- as.data.frame(myMetric(rforest$resample))
fam<- "rforest"
total<- data.frame(metric = mod_info[(mod_info$splitrule == "gini") &
                                         (mod_info$mtry == 8) &
                                         (mod_info$min.node.size== 1),c("metric")],
                   modelo = fam,
                   family = fam)


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
mod_best<- train(frml_All, data=datTrain_sca, metric="ROC",
                 method="ranger",
                 trControl=control, tuneGrid=expand.grid(mtry = 8,
                                                         splitrule = "gini",
                                                         min.node.size = 1),
                 replace = TRUE, num.trees = 500, sample.fraction = 0.8)
test_pred <- as.factor(ifelse(predict(mod_best, datTest_sca) == "X0",0,1))
submission$Survived = test_pred
fwrite(submission, "rforest_sub.csv", row.names = FALSE)
