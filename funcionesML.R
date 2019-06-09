# Imputación de missings --------------------------------------------------

# Rellena los missings en var1 con valores de var1 parecidos a entradas con valores similares de var2
# Inputa cuantitativa mediante cuantitativa
fillMissCont<- function(data, var1, var2, no.lower){
    dat<- as.data.frame(data)
    dat$var2.fac<- cut(dat[[var2]], 8)
    missings<- dat[is.na(dat[[var1]]),c(var1, var2, "var2.fac")]
    not.missings<- dat[!is.na(dat[[var1]]), c(var1, var2, "var2.fac")]
    inputator<- as.data.frame(aggregate(not.missings[,1], list(not.missings$var2.fac), mean))
    inputator$sd<- aggregate(not.missings[,1], list(not.missings$var2.fac), sd)$x
    set.seed(12345)
    if(no.lower){
        aleat<- runif(nrow(missings), 0, 1)
    }else{
        aleat<- runif(nrow(missings), -1, 1)
    }
    for(i in seq(1,nrow(missings))){
        miss.group<- missings$var2.fac[i]
        al<- aleat[i]
        val<- inputator[inputator$Group.1 == miss.group, 2] + al * inputator[inputator$Group.1 == miss.group, 3]
        dat[rownames(missings)[i],var1]<- val
    }
    return(dat[, -ncol(dat)])
}


# Selección de Variables --------------------------------------------------
# Gráfico de la V cde Cramer
Vcramer<-function(v, target){
    if (is.numeric(v)){
        v<-cut(v,5)
    }
    if (is.numeric(target)){
        target<-cut(target,5)
    }
    cramer.v(table(v,target))
}

myStepRepetidoBin<- function(data=data, vardep="vardep",
                               listconti="listconti", sinicio=12345, sfinal=12355,
                               porcen=0.8, criterio="AIC"){
    library(MASS)
    library(dplyr)
    resultados<-data.frame(variables= c(listconti))
    data<-data[,c(listconti,vardep)]
    formu1<-formula(paste("factor(",vardep,")~.",sep=""))
    formu2<-formula(paste("factor(",vardep,")~1",sep=""))
    
    for (semilla in sinicio:sfinal){
        set.seed(semilla)
        sample <- sample.int(n = nrow(data), size = floor(porcen*nrow(data)), replace = F)
        train <- data[sample, ]
        test  <- data[-sample, ]
        # Modelos completo y nulo
        full<-glm(formu1,data=train,family = binomial(link="logit"))
        null<-glm(formu2,data=train,family = binomial(link="logit"))
        if  (criterio=='AIC'){
            selec1<-stepAIC(null,scope=list(upper=full),
                            direction="both", family = binomial(link="logit"), trace=FALSE)
            }
        else if(criterio=='BIC'){
            k1=log(nrow(train))
            selec1<-stepAIC(null,scope=list(upper=full),
                            direction="both",family = binomial(link="logit"), k=k1, trace=FALSE)
        }
        used_vars<- names(get_all_vars(selec1, data))
        used_vars<- used_vars[used_vars != vardep]
        vec<-as.numeric(resultados$variable %in% used_vars)
        resultados[[as.character(semilla)]] <- vec
    }
    # Most used vars
    aux<- rowSums(resultados[,seq(2,ncol(resultados))])
    mostUsed<- as.vector(resultados$variables[which(aux>=(0.6*max(aux)))])
    salida = list(resultados, mostUsed)
}
# Ejemplo myStepRepetidoBin
# load("saheartbis.Rda")
# 
# listconti<-c("sbp", "tobacco", "ldl", "adiposity",
#  "obesity", "alcohol","age", "typea",
#  "famhist.Absent", "famhist.Present")
# vardep<-c("chd")
# 
# data<-saheartbis
# data<-data[,c(listconti,vardep)]
# lista<- myStepRepetidoBin(data=data,
#                           vardep=vardep,listconti=listconti,sinicio=12345,
#                           sfinal=12355,porcen=0.8, criterio="BIC")
# 
# lista[[1]]
# lista[[2]]

# Métrica -------------------------------------------------------------------------------------
myMetric<- function(resample){
    resample$metric<- (1/3)*resample[,c("Sens")] + (2/3)*resample[,c("Spec")]
    return(resample[order(resample$metric),])
}


