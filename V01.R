###########################################################
# Competición Titanic Kaggle
# Algoritmo tipo: REGRESIÓN LOGÍSTICA
# Resultado: 0.79425
# Posición: 2212
# Percentil: 24.1
# Fecha: 15 - 05 - 2019
###########################################################

setwd("~/Desktop/Santi/master_BD/machine_learning/entrega/")
# 0. Cargo paquetes ----
library(ggplot2)
library(data.table)
library(stringr)
library(tidyr)
library(gridExtra)
library(questionr)
library(caret)
library(dplyr)
library(ROCR)

## 0.1 Funciones útiles ----
source("funcionesML.R")

# 1. Cargo los datos ----
train <- data.table(fread("./train.csv"))
test <- data.table(fread("./test.csv"))

# 2. Tratamiento de datos ----
datos<- train
dim(datos)
names(datos)
head(datos, 20)

## 2.1 Reformateo ----

datos$CabLetter <- substr(datos$Cabin, 1, 1)
datos$CabLetter<- as.factor(replace(datos$CabLetter,
                                    which(datos$CabLetter == "" | datos$CabLetter == "T"), "U"))
datos$CabNumber <- as.numeric(stringr::str_extract(datos$Cabin, "\\d+"))
table(is.na(datos$CabNumber))
datos$CabNumber_bin<- ifelse(is.na(datos$CabNumber), 0, 1)

datos$TickLetters <- stringr::str_extract(datos$Ticket, "[A-Za-z]+")
datos$TickLetters <- as.factor(replace(datos$TickLetters, which(is.na(datos$TickLetters)), "Unk"))
table(datos$TickLetters)
datos$TickNumber <- as.numeric(stringr::str_extract(datos$Ticket, "[^A-Za-z/]\\d+"))
table(is.na(datos$TickNumber))
datos$TickNumber<- replace(datos$TickNumber, which(is.na(datos$TickNumber)), 0)

# El nombre tiene bastante información, primero el título
datos$Title <- substring(datos$Name, regexpr(",",datos$Name)+2,
                         regexpr("\\.",datos$Name)-1)
datos$Title[datos$Title %in% c("Capt","Don","Major","Col","Rev","Dr","Sir","Jonkheer")] <- "arist-man"
datos$Title[datos$Title %in% c("Dona","the Countess","Mme","Mlle","Lady")] <- "arist-woman"
datos$Title[datos$Title %in% c("Mr")] <- "man"
datos$Title[datos$Title %in% c("Mrs", "Ms", "Miss")] <- "woman"
datos$Title[datos$Title %in% c("Master")] <- "boy"
# El apellido
datos$Surname <- substring(datos$Name,0,regexpr(",",datos$Name)-1)
datos$SurnFreq <- ave(1:nrow(datos), datos$Surname, FUN=length)
datos$SurnSurvival <- ave(datos$Survived, datos$Surname)
# Miramos si es coherente
condicion<- ((datos$SurnSurvival>0.5 & datos$Survived!=1) | (datos$SurnSurvival<=0.5 & datos$Survived==1))
length(datos$Name[condicion])/length(datos$Name)
# Solo un 7.1% de los pasajeros siguió un destino distinto al de aquellos con su mismo apellido
length(datos$Name[(datos$SurnFreq>1 & condicion)])/length(datos$Name[(datos$SurnFreq>1)])
# Un 17.6% de entre aquellos cuyo grupo era multitudinario

## 2.2 Inputo missings ----
# Vemos la proporción de missings en cada varaible
for(var in names(datos)){
    print(var)
    print(table(is.na(datos[[var]])))
}
# Solo hay missings en CabNumber, en Age y en Embarked. Los imputo.
length(datos$Name[is.na(datos$Age)])/length(datos$Name)
summary(datos$Age)
datos<- fillMissCont(datos, "Age", "Fare", FALSE)
summary(datos$Age)# No ha cambiado mucho la distribución de edad
# Inputo las filas cuyo lugar de a bordo se desconoce
datos$Surname[datos$Embarked == ""]
datos[datos$Surname == "Icard",]
datos[datos$Surname == "Stone",]
# Dos mujeres mayores que viajaban juntas con el mismo identificador de cabina
datos %>%
    filter(CabLetter == "B" & TickLetters == "Unk") %>%
    group_by(Embarked) %>% 
    summarise(n = n()) %>%
    filter(n == max(n))
datos[datos$Surname == "Stone",c("Embarked")] = "S"
datos[datos$Surname == "Icard",c("Embarked")] = "S"
# Les asigno el puerto más frecuente en su estilo
length(datos$Name[is.na(datos$CabNumber)])/length(datos$Name)
length(datos$Cabin[which(is.na(datos$CabNumber) & datos$CabLetter == "U")])
# Nada que hacer, esta variable la eliminaremos
## 2.3 Variables adicionales ----
# Incluyo la variable FamilySize que servirá de soporte a SurnFreq
datos$FamilySize<- datos$Parch + datos$SibSp

## 2.4 Agrupación de categorías ----
# Defino aquellas variables que son factores
names(datos)
datos$Survived<- as.factor(datos$Survived)
datos$Pclass<- as.factor(datos$Pclass)
datos$Sex<- as.factor(datos$Sex)
datos$Embarked<- droplevels(as.factor(datos$Embarked))
datos$CabLetter<- as.factor(datos$CabLetter)
datos$CabNumber_bin<- as.factor(datos$CabNumber_bin)
datos$TickLetters<- as.factor(datos$TickLetters)
datos$Title<- as.factor(datos$Title)
datos$Surname<- as.factor(datos$Surname)

str(datos) # Solo TickLetter tiene más de 10 categorías
# Agrupo TickLetter aquellos que se distribuyan de forma parecida
ggplot(data=datos[datos$TickLetters != "Unk",],
       aes(x=TickLetters, fill=Survived)) +
    geom_bar(stat="count", color="black") + #position=position_dodge())+
    theme_minimal() + coord_flip()
aux<- datos %>%
    dplyr::select(Survived, TickLetters) %>% 
    group_by(TickLetters, Survived) %>% 
    summarise(n = n()) %>%
    mutate(freq = n / sum(n))
TickLet<- c("SW", "SO", "PP", "F", "PC", "SC", "WE", "P", "C", "STON", "Unk", "LINE", "S", "SOTON",
            "W", "CA", "A", "Fa", "SCO")
aux$TickLetters<- factor(aux$TickLetters, levels=TickLet)
p1<- ggplot(data=aux,
            aes(x=TickLetters, y=freq, fill=Survived)) +
    geom_bar(stat="identity", color="black", show.legend = FALSE)+
    theme_minimal() + coord_flip()

aux2<- aux[(aux$Survived==1),]
aux2$freq_fac<- cut(aux2$freq, 5)
aux3<- aux[(aux$Survived==0),]
aux3$freq_fac<- cut(aux3$freq, 5)
class<- merge(aux2[,c(1,5)], aux3[,c(1,5)], by="TickLetters", all = TRUE)
class[order(-class[,2], class[,3]),]
# Hago los grupos con cuidado mirando esta tabla
# upup_sup = c("F", "PC", "PP", "SO", "SW")
# up_sup = c("C", "P", "STON", "WE", "SC")
# mid_sup = c("Unk")
# low_sup = c("LINE", "S")
# lowlow_sup = c("SCO", "SOTON", "W", "A", "Fa", "CA")

datos$TickLetters_g<- ""
datos$TickLetters_g[datos$TickLetters %in% c("SCO", "SOTON", "W", "A", "Fa", "CA")] <- "lowlow_sur"
datos$TickLetters_g[datos$TickLetters %in% c("LINE", "S")] <- "low_sur"
datos$TickLetters_g[datos$TickLetters %in% c("Unk")] <- "mid_sur"
datos$TickLetters_g[datos$TickLetters %in% c("C", "P", "STON", "WE", "SC")] <- "up_sur"
datos$TickLetters_g[datos$TickLetters %in% c("F", "PC", "PP", "SO", "SW")] <- "upup_sur"
datos$TickLetters_g<- droplevels(factor(datos$TickLetters_g, levels = c("", "upup_sur", "up_sur",
                                                                        "mid_sur", "low_sur", "lowlow_sur")))
aux<- datos %>%
    dplyr::select(Survived, TickLetters_g) %>% 
    group_by(TickLetters_g, Survived) %>% 
    summarise(n = n()) %>%
    mutate(freq = n / sum(n))
p2<- ggplot(data=aux, 
            aes(x=TickLetters_g, y=freq, fill=Survived)) +
    geom_bar(stat="identity", color="black") + # position=position_dodge())+
    theme_minimal() + coord_flip()
gridExtra::grid.arrange(p1, p2, ncol=2, nrow=1)

### 2.41 Miro algunas variables más ----
p1<- datos %>%
    dplyr::select(Embarked, Survived) %>% 
    group_by(Embarked, Survived) %>% 
    summarise(n = n()) %>%
    mutate(freq = n / sum(n))%>%
    ggplot(aes(x=Embarked, y=freq, fill=Survived)) +
    geom_bar(stat="identity", color="black") + # position=position_dodge())+
    theme_minimal() + coord_flip()
p2<- datos %>%
    dplyr::select(Pclass, Survived) %>% 
    group_by(Pclass, Survived) %>% 
    summarise(n = n()) %>%
    mutate(freq = n / sum(n))%>%
    ggplot(aes(x=Pclass, y=freq, fill=Survived)) +
    geom_bar(stat="identity", color="black") + # position=position_dodge())+
    theme_minimal() + coord_flip()
p3<- datos %>%
    dplyr::select(Sex, Survived) %>% 
    group_by(Sex, Survived) %>% 
    summarise(n = n()) %>%
    mutate(freq = n / sum(n))%>%
    ggplot(aes(x=Sex, y=freq, fill=Survived)) +
    geom_bar(stat="identity", color="black") + # position=position_dodge())+
    theme_minimal() + coord_flip()
p4<- datos %>%
    dplyr::select(Title, Survived) %>% 
    group_by(Title, Survived) %>% 
    summarise(n = n()) %>%
    mutate(freq = n / sum(n))%>%
    ggplot(aes(x=Title, y=freq, fill=Survived)) +
    geom_bar(stat="identity", color="black") + # position=position_dodge())+
    theme_minimal() + coord_flip()
gridExtra::grid.arrange(p1, p2, p3, p4, ncol=2, nrow=2)

## 2.5 Eliminación de variables inútiles ----
str(datos)
dput(names(datos))
rownames(datos)<- datos$PassengerId
datos<- dplyr::select(datos, c("Survived", "Pclass", "Sex", "Age", "SibSp", "Parch",
                        "Fare", "Embarked", "CabLetter", "CabNumber_bin",
                        "TickNumber", "Title", "SurnFreq", "FamilySize",
                        "TickLetters_g"))
set.seed(1234)
datos$aleatorio<- rnorm(nrow(datos))
saveRDS(datos, "train_clean")

## 2.6 Repetimos las transformaciones para test ----
datos<- test
### 2.61 Camarote ----
datos$CabLetter <- substr(datos$Cabin, 1, 1)
datos$CabLetter<- as.factor(replace(datos$CabLetter,
                                    which(datos$CabLetter == "" | datos$CabLetter == "T"), "U"))
datos$CabNumber <- as.numeric(stringr::str_extract(datos$Cabin, "\\d+"))
table(is.na(datos$CabNumber))
datos$CabNumber_bin<- ifelse(is.na(datos$CabNumber), 0, 1)
### 2.62 Ticket ----

datos$TickLetters <- stringr::str_extract(datos$Ticket, "[A-Za-z]+")
datos$TickLetters <- as.factor(replace(datos$TickLetters, which(is.na(datos$TickLetters) | !(datos$TickLetters %in% TickLet)), "Unk"))
table(datos$TickLetters)
datos$TickNumber <- as.numeric(stringr::str_extract(datos$Ticket, "[^A-Za-z/]\\d+"))
table(is.na(datos$TickNumber))
datos$TickNumber<- replace(datos$TickNumber, which(is.na(datos$TickNumber)), 0)
datos$Title <- substring(datos$Name, regexpr(",",datos$Name)+2,
                         regexpr("\\.",datos$Name)-1)
### 2.63 Nombre ----
datos$Title[datos$Title %in% c("Capt","Don","Major","Col","Rev","Dr","Sir","Jonkheer")] <- "arist-man"
datos$Title[datos$Title %in% c("Dona","the Countess","Mme","Mlle","Lady")] <- "arist-woman"
datos$Title[datos$Title %in% c("Mr")] <- "man"
datos$Title[datos$Title %in% c("Mrs", "Ms", "Miss")] <- "woman"
datos$Title[datos$Title %in% c("Master")] <- "boy"
datos$Surname <- substring(datos$Name,0,regexpr(",",datos$Name)-1)
datos$SurnFreq <- ave(1:nrow(datos), datos$Surname, FUN=length)
### 2.64 Missings ----
for(var in names(datos)){
    print(var)
    print(table(is.na(datos[[var]])))
}
### 2.65 Fare ----
datos[is.na(Fare),c("Fare")]<- as.numeric(datos %>%
                                              filter(Pclass == as.numeric(datos[is.na(Fare),c("Pclass")])) %>%
                                              summarise(media = mean(Fare, na.rm=T))
)
### 2.66 Edad ----
length(datos$Name[is.na(datos$Age)])/length(datos$Name)
summary(datos$Age)
datos<- fillMissCont(datos, "Age", "Fare", FALSE)
summary(datos$Age)# No ha cambiado mucho la distribución de edad
### 2.67 FamilySize ----
datos$FamilySize<- datos$Parch + datos$SibSp
names(datos)
datos$Pclass<- as.factor(datos$Pclass)
datos$Sex<- as.factor(datos$Sex)
datos$Embarked<- droplevels(as.factor(datos$Embarked))
datos$CabLetter<- as.factor(datos$CabLetter)
datos$CabNumber_bin<- as.factor(datos$CabNumber_bin)
datos$TickLetters<- as.factor(datos$TickLetters)
datos$Title<- as.factor(datos$Title)
datos$Surname<- as.factor(datos$Surname)
### 2.68 TickLetters_grouped ----
datos$TickLetters_g<- ""
datos$TickLetters_g[datos$TickLetters %in% c("SCO", "SOTON", "W", "A", "Fa", "CA")] <- "lowlow_sur"
datos$TickLetters_g[datos$TickLetters %in% c("LINE", "S")] <- "low_sur"
datos$TickLetters_g[datos$TickLetters %in% c("Unk")] <- "mid_sur"
datos$TickLetters_g[datos$TickLetters %in% c("C", "P", "STON", "WE", "SC")] <- "up_sur"
datos$TickLetters_g[datos$TickLetters %in% c("F", "PC", "PP", "SO", "SW")] <- "upup_sur"
datos$TickLetters_g<- droplevels(factor(datos$TickLetters_g, levels = c("", "upup_sur", "up_sur",
                                                                        "mid_sur", "low_sur", "lowlow_sur")))
### 2.69 Save the data ----
rownames(datos)<- datos$PassengerId
datos<- dplyr::select(datos, c("Pclass", "Sex", "Age", "SibSp", "Parch",
                        "Fare", "Embarked", "CabLetter", "CabNumber_bin",
                        "TickNumber", "Title", "SurnFreq", "FamilySize",
                        "TickLetters_g"))
set.seed(1234)
datos$aleatorio<- rnorm(nrow(datos))
saveRDS(datos, "test_clean")

# 3. Modelado ----
datTrain<- readRDS("train_clean")
datTest<- readRDS("test_clean")
probTrain<- data.frame(PassengerId = as.numeric(row.names(datTrain)), y=datTrain$Survived)
probTest<- data.frame(PassengerId = as.numeric(row.names(datTest)))
submission<- data.frame(PassengerId = as.numeric(row.names(datTest)))

## 3.1 Selección de variables ----
y_var<- "Survived"
sel_All<- c("Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", 
            "Embarked", "CabLetter", "CabNumber_bin", "TickNumber", "Title", 
            "SurnFreq", "FamilySize", "TickLetters_g", "aleatorio")

# Criterio de Cramer
VC<-data.frame(VC=sapply(datTrain, function(x) Vcramer(x,datTrain$Survived)))
VC[c("Fare"),]<- cramer.v(table(datTrain$Fare, datTrain$Survived))
VC[c("TickNumber"),]<- cramer.v(table(datTrain$TickNumber, datTrain$Survived))
VC$id = factor(row.names(VC), levels=row.names(VC)[order(VC$VC)])
ggplot(data=VC, aes(x=id, y=VC))+
    geom_bar(stat="identity", color="black", fill="lightblue2", show.legend = FALSE, alpha=0.5) +
    theme_minimal() + coord_flip()
sel_VCr<- c("Sex", "Fare", "TickNumber", "Title")

# Criterios AIC y BIC
lista<- myStepRepetidoBin(data=datTrain,
                          vardep=y_var, listconti=sel_All, sinicio=12345,
                          sfinal=12365, porcen=0.8, criterio="AIC")
dput(lista[[2]])
sel_AIC<- c("Pclass", "Age", "CabNumber_bin", "Title", "FamilySize", "TickLetters_g")
lista<- myStepRepetidoBin(data=datTrain,
                          vardep=y_var, listconti=sel_All, sinicio=12345,
                          sfinal=12365, porcen=0.8, criterio="BIC")
dput(lista[[2]])
sel_BIC<- c("Pclass", "Title")

# Defino los modelos
frml_All<- as.formula(paste(y_var, paste(sel_All, collapse="+"), sep="~"))
frml_VCr<- as.formula(paste(y_var, paste(sel_VCr, collapse="+"), sep="~"))
frml_AIC<- as.formula(paste(y_var, paste(sel_AIC, collapse="+"), sep="~"))
frml_BIC<- as.formula(paste(y_var, paste(sel_BIC, collapse="+"), sep="~"))
mod_nul<- glm(Survived ~ 1, family = binomial(link="logit"), data = datTrain)
mod_ful<- glm(frml_All, family = binomial(link="logit"), data = datTrain)
mod_VCr<- glm(frml_VCr, family = binomial(link="logit"), data = datTrain)
mod_AIC<- glm(frml_AIC, family = binomial(link="logit"), data = datTrain)
mod_BIC<- glm(frml_BIC, family = binomial(link="logit"), data = datTrain)

## 3.2 Validación cruzada ----
auxVarObj<-datTrain$Survived
datTrain$Survived<-make.names(datTrain$Survived) #formateo la variable objetivo para que funcione el codigo
modelos.nombres = c("glm_full", "glm_Cra", "glm_AIC", "glm_BIC")
modelos.formula<-sapply(frml_full, frml_VCr, frml_AIC, frml_BIC)
total<- c()
for (i in 1:length(modelos.nombres)){
    set.seed(1712)
    vcr<-train(modelos.formula[[i]], data = datTrain,
               method = "glm", family="binomial", metric = "ROC",
               trControl = trainControl(method="repeatedcv", number=8, repeats=25,
                                        summaryFunction=twoClassSummary,
                                        classProbs=TRUE,returnResamp="all")
    )
    total<-rbind(total,data.frame(roc=vcr$resample[,1], modelo=rep(modelos.nombres[i],
                                                                   nrow(vcr$resample))))
    saveRDS(vcr, paste("./mod/", modelos.nombres[i], sep=""))
}

modelos<- list(mod_ful, mod_VCr, mod_AIC, mod_BIC)
comparativa<- data.frame(row.names = unique(total$modelo))
comparativa$mean_ROC<- aggregate(roc~modelo, data = total, mean)$roc
comparativa$std_ROC<- aggregate(roc~modelo, data = total, sd)$roc
comparativa$variables<- sapply(modelos, function (x){length(coef(x))})
comparativa<- comparativa[order(comparativa$mean_ROC),]
total$family<- "glm"
saveRDS(total, "./tab/cv_all")
saveRDS(comparativa, "./tab/glm_compared")
ggplot(total, aes(x=modelo, y=roc, fill=family)) + 
    geom_boxplot(outlier.colour="red", outlier.shape=20,
                 outlier.size=2) +
    stat_summary(fun.y=mean, geom="point", colour="white",fill="grey", shape=21, size=3) +
    scale_x_discrete(limits=as.vector(unique(total$modelo)[order(aggregate(roc~modelo, data=total, mean)$roc)])) + 
    theme_minimal() + theme_update(plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
                                   axis.text.x = element_text(size = 10, face = "bold", angle = 45)) +
    ggtitle("Área bajo la curva ROC")

# predicted proability
for(i in seq(1, length(modelos))){
    probTest[[modelos.nombres[i]]]<- predict(modelos[[i]], type='response', datTest)
    probTrain[[modelos.nombres[i]]]<- predict(modelos[[i]], type='response', datTrain)
    saveRDS(modelos[[i]], paste("./mod/",modelos.nombres[i], sep=""))
}

# 4. Make a Submission with the best model ----
train_pred <- predict(mod_AIC, type='response', datTrain) # type='response', Para predecir probabilidad
test_pred <- predict(mod_AIC, type='response', datTest)
datTrain$Survived<- auxVarObj
ROC_met<- prediction(train_pred, datTrain$Survived)
ROC_per = performance(ROC_met, "tpr", "fpr")
plot(ROC_per, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7), main="Curva ROC glmAIC")
# Select the best probability, to split survived from not survived
accs<- seq(0,4)
cutoffs<- seq(0, 1, length.out = 5)
while(any(accs != accs[1])){
    accs<- c()
    for(cutoff in cutoffs){
        aux<- ifelse(train_pred<cutoff, 0, 1)
        acc<- sum(aux==datTrain$Survived)/nrow(datTrain)
        accs<- c(accs, acc)
    }
    low_lim<- cutoffs[max(which(accs==max(accs)))-1]
    upp_lim<- cutoffs[min(which(accs==max(accs)))+1]
    cutoffs<- seq(low_lim, upp_lim, length.out = 5)
}
cutoff<- mean(cutoffs)
submission$Survived = ifelse(test_pred<cutoff, 0, 1)
write.csv(submission, "glm_sub.csv", row.names = FALSE)
