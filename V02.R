###########################################################
# Competición Titanic Kaggle
# Algoritmo tipo: RED NEURONAL
# Resultado: 0.77511
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

# 1. Carga de datos ----
datTrain<- readRDS("train_clean")
datTest<- readRDS("test_clean")

# 1.1 Escalo los datos ----
num_sca<- as.data.frame(scale(Filter(is.numeric, datTrain)))
not_num<- datTrain[,!(names(datTrain) %in% names(num_sca))]
datTrain_sca<- cbind(not_num, num_sca)
saveRDS(datTrain_sca, "train_clean_sca")
num_sca<- as.data.frame(scale(Filter(is.numeric, datTest)))
not_num<- datTest[,!(names(datTest) %in% names(num_sca))]
datTest_sca<- cbind(not_num, num_sca)                          
saveRDS(datTest_sca, "test_clean_sca")

# 3. Modelado ----
submission<- data.frame(PassengerId = as.numeric(row.names(datTest)))

## 3.1 Selección de variables ----
y_var<- "Survived"
sel_All<- c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", 
            "Embarked", "CabLetter", "CabNumber_bin", "TickNumber", "Title", 
            "SurnFreq", "FamilySize", "TickLetters_g")
sel_VCr<- c("Sex", "Fare", "TickNumber", "Title")
sel_AIC<- c("Pclass", "Age", "CabNumber_bin", "Title", "FamilySize", "TickLetters_g")
sel_BIC<- c("Pclass", "Title")

# Fórmulas
frml_All<- as.formula(paste(y_var, paste(sel_All, collapse="+"), sep="~"))
frml_Vcr<- as.formula(paste(y_var, paste(sel_VCr, collapse="+"), sep="~"))
frml_AIC<- as.formula(paste(y_var, paste(sel_AIC, collapse="+"), sep="~"))
frml_BIC<- as.formula(paste(y_var, paste(sel_BIC, collapse="+"), sep="~"))

# 3.2 Tuneo hiperparámetros ----
datTrain_sca$Survived<-make.names(datTrain_sca$Survived)
control<-trainControl(method = "LGOCV", p=0.8, number=5, savePredictions = "all",
                      summaryFunction=twoClassSummary, classProbs=TRUE, returnResamp="all",
                      verboseIter=FALSE)

nnetgrid <-  expand.grid(size=c(3,4,5,6,7),decay=seq(0.03, 0.05, length.out=10), bag=F)
rednnet<- train(frml_All, data=datTrain_sca, metric="ROC",
                method="avNNet", linout=TRUE, maxit=120,
                trControl=control, repeats=5, tuneGrid=nnetgrid)
datos<- as.data.frame(rednnet$results)[order(rednnet$resample),]
saveRDS(datos, "./tab/avNNet_tune")
datos$decay<- sapply(datos$decay, function(x){round(x,3)})
ggplot(datos, aes(size, ROC, fill=ROCSD)) + geom_point(shape=21, size=2) +
    facet_grid(cols = vars(decay)) +
    ggtitle("Área bajo la curva ROC para cada configuración") +
    scale_y_continuous(labels = function(x){format(round(x, 2), nsmall=2)})


# 3.3 Validación cruzada ----
nnetgrid <-  expand.grid(size=c(datos$size[which(datos$ROC == max(datos$ROC))]),
                         decay=c(datos$decay[which(datos$ROC == max(datos$ROC))]),
                         bag=F)
modelos.nombres <- c("avNNet_full", "avNNet_Cra", "avNNet_AIC", "avNNet_BIC")
modelos.formula<- c(frml_All, frml_Vcr, frml_AIC, frml_BIC)
total<- c()
for (i in 1:length(modelos.nombres)){
    set.seed(1712)
    vcr<-?train(as.formula(modelos.formula[[i]]), data = datTrain_sca,
               method="avNNet", linout=TRUE, maxit=120,
               tuneGrid=nnetgrid, metric="ROC",
               trControl = trainControl(method = "repeatedcv",
                                        number=8, repeats=10,
                                        savePredictions = "all",
                                        summaryFunction=twoClassSummary,
                                        classProbs=TRUE, returnResamp="all",
                                        verboseIter=FALSE)
    )
    total<-rbind(total,data.frame(roc=vcr$resample[,1], modelo=rep(modelos.nombres[i],
                                                                   nrow(vcr$resample)))) 
    saveRDS(vcr, paste("./mod/", modelos.nombres[i], sep=""))
}
comparativa<- data.frame(row.names = unique(total$modelo))
comparativa$mean_ROC<- aggregate(roc~modelo, data = total, mean)$roc
comparativa$std_ROC<- aggregate(roc~modelo, data = total, sd)$roc
comparativa<- comparativa[order(comparativa$mean_ROC),]
saveRDS(comparativa, "./tab/avNNet_compared")

# 3.4 Boxplot ----
cv_all<- readRDS("./tab/cv_all")
total$family<- "avNNet"
cv_all<- rbind(cv_all, total)
ggplot(cv_all, aes(x=modelo, y=roc, fill=family)) + 
    geom_boxplot(outlier.colour="red", outlier.shape=20,
                 outlier.size=2) +
    stat_summary(fun.y=mean, geom="point", colour="white",fill="grey", shape=21, size=3) +
    scale_x_discrete(limits=as.vector(unique(cv_all$modelo)[order(aggregate(roc~modelo, data=cv_all, mean)$roc)])) + 
    theme_minimal() + theme_update(plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
                                   axis.text.x = element_text(size = 10, face = "bold", angle = 45,
                                                              hjust = 1)) +
    ggtitle("Área bajo la curva ROC")
saveRDS(cv_all, "./tab/cv_all")

# 4. Make a Submission with the best model ----
mod_best<- readRDS("./mod/avNNet_full")
test_pred <- as.factor(ifelse(predict(mod_best, datTest_sca) == "X0",0,1))
submission$Survived = test_pred
write.csv(submission, "avNNet_sub.csv", row.names = FALSE)
