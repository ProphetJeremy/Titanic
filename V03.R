###########################################################
# Competición Titanic Kaggle
# Algoritmo tipo: ÁRBOL DE DECISIÓN
# Resultado: 0.75598
# Posición: No mejora
# Percentil: No mejora 
# Fecha: 18 - 05 - 2019
###########################################################

setwd("~/Desktop/Santi/master_BD/machine_learning/entrega/")
setwd("/cloud/project/entrega/")
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
library(rpart)
library(rpart.plot)

# 1. Carga de datos ----
datTrain<- readRDS("train_clean")
datTest<- readRDS("test_clean")

# 3. Modelado ----
submission<- data.frame(PassengerId = as.numeric(rownames(datTest)))

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
datTrain$Survived<-make.names(datTrain$Survived)
minbuckets = seq(1,20)
for(minb_val in minbuckets){
    arbol<- rpart(frml_All, data = datTrain,
                  minbucket=minb_val, method = "class", parms=list(split="gini"), cp=0)
    plotcp(arbol)
}

# Eligo min_val 8 y cp = 0.025
arbol<- rpart(frml_All, data = datTrain,
              minbucket=4, method = "class", parms=list(split="gini"), cp=0.02)
rpart.plot(arbol,extra=1)
aux<- data.frame(id = names(arbol$variable.importance),
                 val = arbol$variable.importance)
ggplot(data = aux, aes(x=id, y=val)) +
    geom_bar(stat="identity", fill="steelblue") +
    scale_x_discrete(limits=as.vector(aux$id[order(aux$val)])) +
    theme_minimal() +
    theme_update(plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
                 axis.text.x = element_text(size = 10, face = "bold", angle = 45, hjust = 1)) + 
    ggtitle("Importancia de las variables según el árbol")

# sel_tree = c("Title", "Sex", "FamilySize", "Parch", "SibSp", "Pclass", "Age", "Fare")

# 3.3 Validación cruzada ----
arbolgrid<- expand.grid(cp=c(0,0.001,0.01,0.05,0.1))
modelos.nombres <- c("tree_full", "tree_Cra", "tree_AIC", "tree_BIC", "tree_Tre")
modelos.formula<- c(frml_All, frml_Vcr, frml_AIC, frml_BIC, frml_Tre)
total<- c()
for (i in 1:length(modelos.nombres)){
    set.seed(1712)
    vcr<-train(as.formula(modelos.formula[[i]]), data = datTrain,
               method="rpart", minbucket=8,
               tuneGrid=arbolgrid, metric="ROC",
               trControl = trainControl(method = "repeatedcv",
                                        number=8, repeats=10,
                                        savePredictions = "all",
                                        summaryFunction=twoClassSummary,
                                        classProbs=TRUE, returnResamp="all",
                                        verboseIter=FALSE)
    )
    total<-rbind(total,data.frame(roc=vcr$resample[,2], modelo=rep(modelos.nombres[i],
                                                                   nrow(vcr$resample))))
    probTest[[modelos.nombres[i]]]<- as.factor(ifelse(predict(vcr, datTest) == "X0",0,1))
    probTrain[[modelos.nombres[i]]]<- as.factor(ifelse(predict(vcr, datTrain) == "X0",0,1))
    saveRDS(vcr, paste("./mod/", modelos.nombres[i], sep=""))
}
write.csv(probTrain, "./tab/probabilidadTrain.csv", row.names = FALSE)
write.csv(probTest, "./tab/probabilidadTest.csv", row.names = FALSE)

comparativa<- data.frame(row.names = unique(total$modelo))
comparativa$mean_ROC<- aggregate(roc~modelo, data = total, mean)$roc
comparativa$std_ROC<- aggregate(roc~modelo, data = total, sd)$roc
comparativa<- comparativa[order(comparativa$mean_ROC),]
saveRDS(comparativa, "./tab/tree_compared")

# 3.4 Boxplot ----
cv_all<- readRDS("./tab/cv_all")
total$family<- "tree"
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
mod_best<- readRDS("./mod/tree_AIC")
test_pred <- as.factor(ifelse(predict(mod_best, datTest) == "X0",0,1))
submission$Survived = test_pred
write.csv(submission, "tree_sub.csv", row.names = FALSE)
