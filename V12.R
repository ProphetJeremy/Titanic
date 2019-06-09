###########################################################
# Competición Titanic Kaggle
# Algoritmo tipo: NAIVE BAYES
# Resultado: 
# Posición: No mejora
# Percentil: No mejora
# Fecha: 15 - 05 - 2019
###########################################################

setwd("~/Desktop/Santi/master_BD/machine_learning/entrega/")
setwd("C:\\Users/n230104/Desktop/personal/master/entrega/entrega/")
source("funcionesML.R")

# 0. Cargo paquetes ----
library(dplyr)
library(klaR)
library(ggplot2)
library(data.table)
library(tidyr)
library(gridExtra)
library(caret)
library(ggplot2)

# 1. Carga de datos ----
datTrain_sca<- readRDS("train_clean_sca_g")
datTest_sca<- readRDS("test_clean_sca_g")
datTotal_sca<- rbind(datTrain_sca[,c(2:ncol(datTrain_sca))], datTest_sca)

probTrain<- readRDS("./tab/probTrain.csv")
probTest<- readRDS("./tab/probTest.csv")
probTotal<- rbind(probTrain[,c(1,3:ncol(probTrain))], probTest)

datTrain_fin<- cbind(datTrain_sca, probTrain[,c(3:ncol(datTrain_sca))])
datTest_fin<- cbind(datTest_sca, probTest[,c(2:ncol(datTest_sca))])
summary(datTrain_fin)

# Submission
submission<- data.frame(PassengerId = as.numeric(row.names(datTest_sca)))

# 2. Validación cruzada ----
datTrain_fin$Survived<-make.names(datTrain_fin$Survived)
model<-train(Survived~., data = datTrain_fin,
             method="nb",
             metric="ROC",
             trControl = trainControl(method = "repeatedcv",
                                      number=8, repeats=10,
                                      savePredictions = "all",
                                      summaryFunction=twoClassSummary,
                                      classProbs=TRUE, returnResamp="all",
                                      verboseIter=FALSE)
    )
mod_info<- as.data.frame(myMetric(model$resample))
mod_info<- drop_na(mod_info)
saveRDS(model, paste("./mod/", "bayes", sep=""))

# 2.1 Boxplot ----
total<- data.frame(metric = mod_info$metric,
                   ROC = mod_info$ROC,
                   modelo = "bayes",
                   family = "bay")
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
# 3. Make a Submission with the best model ----
test_pred <- as.factor(ifelse(predict(model, datTest_fin) == "X0",0,1))
submission$Survived = test_pred
fwrite(submission, "./sub/bayes_sub.csv", row.names = FALSE)
