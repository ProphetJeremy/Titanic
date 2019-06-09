setwd("~/Desktop/Santi/master_BD/machine_learning/entrega/")
source("funcionesML.R")
library(data.table)
library(ggplot2)

modelos_names<- list.files("./mod")
datTrain<- readRDS("train_clean")
datTest<- readRDS("test_clean")
datTrain_sca<- readRDS("train_clean_sca")
datTest_sca<- readRDS("test_clean_sca")
probTrain<- data.frame(PassengerId = as.numeric(row.names(datTrain)), y=datTrain$Survived)
probTest<- data.frame(PassengerId = as.numeric(row.names(datTest)))
binTrain<- data.frame(PassengerId = as.numeric(row.names(datTrain)), y=datTrain$Survived)
binTest<- data.frame(PassengerId = as.numeric(row.names(datTest)))
pons<- c()
for(name in modelos_names){
    set.seed(1712)
    fam<- sub("^([A-Za-z]+)_.*", "\\1", name, perl=TRUE)
    if(fam == "glm"){pons<- c(pons, 1)}
    else if(fam == "avNNet"){pons<- c(pons,2)}
    else if(fam == "tree"){pons<- c(pons,3)}
    else if(fam == "bagg"){pons<- c(pons, 4)}
    else if(fam == "rforest"){pons<- c(pons, 5)}
    else if(fam == "gbm"){pons<- c(pons, 6)}
    else if(fam == "xgb"){pons<- c(pons, 7)}
    else{pons<- c(pons, 8)}
}
sort_df<- data.frame(modelo=modelos_names, make_order=pons)
sort_df<- sort_df[order(pons),]

total<- c()
old_fam<- sub("^([A-Za-z]+)_.*", "\\1", sort_df$modelo[1], perl=TRUE)
for(name in sort_df$modelo){
    set.seed(1712)
    fam<- sub("^([A-Za-z]+)_.*", "\\1", name, perl=TRUE)
    # Plot (simply useless)
    # if(old_fam != fam){
    #     print("Doing this")
    #     png(paste("img/",old_fam,"s.png", sep=""), width = 1000, height=550)
    #     ggplot(total, aes(x=modelo, y=metric, fill=family)) + 
    #         geom_boxplot(outlier.colour="red", outlier.shape=20,
    #                      outlier.size=2) +
    #         stat_summary(fun.y=mean, geom="point", colour="white",fill="grey", shape=21, size=3) +
    #         scale_x_discrete(limits=as.vector(unique(total$modelo)[order(aggregate(metric~modelo, data=total, mean)$metric)])) + 
    #         theme_minimal() + theme_update(plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
    #                                        axis.text.x = element_text(size = 10, face = "bold", angle = 45,
    #                                                                   hjust = 1)) +
    #         ggtitle("Mi métrica")
    #     dev.off()
    #     # readline("Please press the Enter key to see the next plot if there is one.")
    # }
    cat(fam, ", ", old_fam, ", ", name, "\n")
    old_fam<- fam
    model<- readRDS(paste("./mod/", name, sep=""))
    if(fam %in% c('glm', 'tree')){
        probTrain[[name]]<- predict(model, datTrain, type='prob')$X1
        probTest[[name]]<- predict(model, datTest, type='prob')$X1
        binTrain[[name]]<- as.factor(ifelse(predict(model, datTrain) == "X0",0,1))
        binTest[[name]]<- as.factor(ifelse(predict(model, datTest) == "X0",0,1))
    }else{
        probTrain[[name]]<- predict(model, datTrain_sca, type='prob')$X1
        probTest[[name]]<- predict(model, datTest_sca, type='prob')$X1
        binTrain[[name]]<- as.factor(ifelse(predict(model, datTrain_sca) == "X0",0,1))
        binTest[[name]]<- as.factor(ifelse(predict(model, datTest_sca) == "X0",0,1))
    }
    mod_info<- myMetric(model$resample)
    saveRDS(mod_info, paste("./tab/",name,"_CV", sep=""))
    # Collect data
    to_total<- data.frame(metric=mod_info$metric, ROC = mod_info$ROC, modelo=name, family=fam)
    total<- rbind(total, to_total)
    saveRDS(total, paste("./tab/",name,"_CV", sep=""))
}

rank<- as.vector(unique(total$modelo)[order(aggregate(metric~modelo, data=total, mean)$metric)])
rank_pos<- c()
for(name in sort_df$modelo){
    rank_pos<- c(rank_pos, which(rank==name))
}  
sort_df$rank_pos<- rank_pos
saveRDS(sort_df, "./tab/ranking")

# Last plot
png(paste("img/",old_fam,"s.png", sep=""), width = 1000, height=550)
ggplot(total, aes(x=modelo, y=metric, fill=family)) + 
    geom_boxplot(outlier.colour="red", outlier.shape=20,
                 outlier.size=2) +
    stat_summary(fun.y=mean, geom="point", colour="white",fill="grey", shape=21, size=3) +
    scale_x_discrete(limits=as.vector(unique(total$modelo)[order(aggregate(metric~modelo, data=total, mean)$metric)])) + 
    theme_minimal() + theme_update(plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
                                   axis.text.x = element_text(size = 10, face = "bold", angle = 45,
                                                              hjust = 1)) +
    ggtitle("Área bajo la curva metric")
dev.off()

# Save everything
saveRDS(total, "./tab/cv_all")
saveRDS(probTrain, "./tab/probTrain.csv")
saveRDS(probTest, "./tab/probTest.csv")
saveRDS(binTrain, "./tab/binTrain.csv")
saveRDS(binTest, "./tab/binbTest.csv")
