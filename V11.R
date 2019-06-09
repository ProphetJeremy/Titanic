###########################################################
# Competición Titanic Kaggle
# Algoritmo tipo: ENSAMBLADO
# Resultado-mean: 0.78497
# Resultado-mean_pond: 0.79425
# Resultado-best: 0.77511
# Posición: No mejora
# Percentil: No mejora
# Fecha: 15 - 05 - 2019
###########################################################

setwd("~/Desktop/Santi/master_BD/machine_learning/entrega/")
setwd("/cloud/project/entrega/")
setwd("C:\\Users/n230104/Desktop/personal/master/entrega/entrega/")
source("funcionesML.R")

# 0. Cargo paquetes ----
library(dplyr)
library(reshape2)
library(ggplot2)
library(data.table)
library(stringr)
library(tidyr)
library(gridExtra)
library(questionr)
library(caret)
library(cluster)
library(ggplot2)
library(factoextra)
library(FactoMineR)
library(NbClust)

# 1. Carga de datos ----
datTrain<- readRDS("train_clean")
datTest<- readRDS("test_clean")
datTotal<- rbind(datTrain[,c(2:ncol(datTrain))], datTest)

datTrain_sca<- readRDS("train_clean_sca")
datTest_sca<- readRDS("test_clean_sca")
datTotal_sca<- rbind(datTrain_sca[,c(2:ncol(datTrain_sca))], datTest_sca)

## 1.1. Carga de predicciones ----
probTrain<- readRDS("./tab/probTrain.csv")
probTest<- readRDS("./tab/probTest.csv")
probTotal<- rbind(probTrain[,c(1,3:ncol(probTrain))], probTest)

binTrain<- readRDS("./tab/binTrain.csv")
binTest<- readRDS("./tab/binbTest.csv")
binTotal<- rbind(binTrain[,c(1,3:ncol(binTrain))], binTest)

# Submission
submission<- data.frame(PassengerId = as.numeric(row.names(datTest_sca)))

## 1.2. Ponderación
ranking<- readRDS("./tab/ranking")
total<- readRDS("./tab/cv_all")
metricMean<- aggregate(metric~modelo, data=total, mean)
# Fix ponderation values between 0.3 - 0.7
metricMean$pond<- ((metricMean$metric - min(metricMean$metric))*0.4/(max(metricMean$metric)-min(metricMean$metric)))+0.3
metricMean$pond<- metricMean$pond/sum(metricMean$pond)
# Sort values as they are in names(ProbTotal)
posi<- c()
for(mod in metricMean$modelo){
    posi<- c(posi, which(names(probTotal[,c(2:ncol(probTotal))]) == mod))
}
metricMean$posi<- posi
metricMean[order(metricMean$posi),]

# 2. Average predictions ----
# Aggregate the mean and pondered mean probability for each sample
probTotal$mean<- rowMeans(probTotal[,c(2:ncol(probTotal))])
probTotal$mean_pond<- rowSums(metricMean$pond * probTotal[,c(2:(ncol(probTotal)-1))])
# Reescaling
probTotal$mean_pond<- (probTotal$mean_pond - min(probTotal$mean_pond))/(max(probTotal$mean_pond) - min(probTotal$mean_pond))

binTotal_man<- binTotal
for(col in names(binTotal[,c(2:ncol(binTotal))])){
    binTotal_man[[col]]<- as.numeric(ifelse(binTotal_man[[col]]==0,0,1))
}
binTotal$mean<- as.factor(ifelse(rowMeans(binTotal_man[,c(2:ncol(binTotal_man))]) < 0.5,0,1))
binTotal$mean_pond<- as.factor(ifelse(rowSums(metricMean$pond * binTotal_man[,c(2:ncol(binTotal_man))]) < 0.5,0,1))

## 2.1 Doubts ----
# Calculate how many samples will switch if we use a different way to evaluate it
doubts<- c()
for(i in seq(1,nrow(binTotal))){
    a<- all(as.numeric(binTotal[i,c(2:ncol(binTotal))]) == as.numeric(binTotal[i,2]))
    doubts<- c(doubts, ifelse(a, 0, 1))
}
sum(doubts[1:nrow(probTrain)])/nrow(probTrain)
sum(doubts[(nrow(probTrain)+1):nrow(probTotal)])/nrow(probTest)

## 2.2 Mean submissions ----
Sub_mean<- submission
Sub_mean$Survived<- binTotal$mean[(nrow(probTrain)+1):nrow(probTotal)]
fwrite(Sub_mean, "./sub/mean_sub.csv", row.names = FALSE)
Sub_mean_pond<- submission
Sub_mean_pond$Survived<- binTotal$mean_pond[(nrow(probTrain)+1):nrow(probTotal)]
fwrite(Sub_mean_pond, "./sub/mean-pond_sub.csv", row.names = FALSE)

# 3. Heatmap ----
hm_df<- data.frame(PassengerID = row.names(datTrain))
for (col in names(binTotal[,2:ncol(binTotal)])){
    binTotal_sel<- as.numeric(binTotal[c(1:nrow(datTrain)), c(col)])
    hm_df[[col]]<- as.factor(ifelse(binTotal_sel == as.numeric(datTrain$Survived), "HIT", "FAIL"))
}
hm<- melt(hm_df, measure.vars = names(hm_df[,c(2:ncol(hm_df))]), factorsAsStrings = FALSE)
ggplot(data = hm, aes(x = PassengerID, y = variable, fill= value)) +
    geom_tile() +
    ggtitle("Models over Train set") +
    scale_fill_manual(values=c("#ce3939", "#86cc7a"), breaks=levels(hm$value)) + 
    theme_update(plot.title = element_text(size = 12, face = "bold", hjust = 0.5),
                 axis.text.x = element_blank(),
                 axis.title.y = element_blank(),
                 axis.ticks.x = element_blank())

## 3.1 Weird Cases ----
weird_ids<- as.numeric(hm_df[which(hm_df$bagg_full == "FAIL"),]$PassengerID)
weird_pass<- datTrain[which(rownames(datTrain) %in% weird_ids),]
saveRDS(weird_pass, "./tab/weirdPass")

# 4. Clustering ----
# Remove factors
num_cols<- names(which(sapply(datTotal_sca, is.numeric) == TRUE))
fac_cols<- names(datTotal_sca)[!(names(datTotal_sca) %in% num_cols)]

# Join all the measurable data
fullSet<- cbind(datTotal_sca[,num_cols], probTotal[, c(2:ncol(probTotal))])
summary(datTotal_sca)

# Distancias ----
distances <- dist(fullSet, method = "euclidean")
fviz_dist(distances, show_labels = FALSE)

## 4.1 Selección de número de clústers ----
# Elbow method
p1<- fviz_nbclust(fullSet, kmeans, method = "wss") +
    geom_vline(xintercept = 3, linetype = 2)+
    geom_vline(xintercept = 7, linetype = 2)+
    labs(subtitle = "Elbow method")
# Silhouette method
p2<- fviz_nbclust(fullSet, kmeans, method = "silhouette")+
    labs(subtitle = "Silhouette method")
grid.arrange(p1, p2, ncol=2)

## 5. K-Meansn ----
i = 5
km.res <- kmeans(fullSet ,i)
p<- fviz_cluster(km.res, fullSet,
             palette = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07", "#cc71d6", "#8c0000", 
                         "#9b5726", "#a30488", "#016384", "#a168e2"),
             ellipse.type = "convex", # Concentration ellipse
             geom = c("point"),
             repel = TRUE, # Avoid label overplotting (slow)
             show.clust.cent = TRUE, ggtheme = theme_minimal(),
             main = paste("Clustering ",i, "-Means", sep="")
)
grid.arrange(p3, p4, p5, p6, p7, ncol=3)

# 6. Hierarchy clustering ----
res.hc <- hclust(distances, method="ward.D2") 
fviz_dend(res.hc, k = 5, # Cut in seven groups
          cex = 0.5, # label size
          k_colors = c("#2E9FDF", "#E7B800", "#FC4E07", "#8c0000", "#a168e2"),
          color_labels_by_k = TRUE, # color labels by groups
          rect = FALSE, # Add rectangle around groups
          main = "Dendograma"
)
i = 5 # Change this manually to generate multiple plots
grp <- cutree(res.hc, k = i)
p<- fviz_cluster(list(data = fullSet, cluster = grp),
                  palette = c("#2E9FDF", "#00AFBB", "#E7B800", "#FC4E07", "#cc71d6", "#8c0000", 
                              "#9b5726", "#a30488", "#016384", "#a168e2"),
                  ellipse.type = "convex", # Concentration ellipse
                  geom = c("point"),
                  repel = TRUE, # Avoid label overplotting (slow)
                  show.clust.cent = TRUE, ggtheme = theme_minimal(),
                  main = paste("Clustering ",i, "-Jerárquico", sep="")
)
grid.arrange(p3, p4, p5, p6, p7, ncol=3)

# 7. Define sample groups ----
datTotal$group<- as.factor(grp)
datTrain$group<- datTotal$group[c(1:nrow(datTrain))]
datTest$group<- datTotal$group[c((nrow(datTrain)+1):nrow(datTotal))]

datTotal_sca$group<- as.factor(grp)
datTrain_sca$group<- datTotal_sca$group[c(1:nrow(datTrain))]
datTest_sca$group<- datTotal_sca$group[c((nrow(datTrain)+1):nrow(datTotal))]

datFIN<- cbind(datTotal, probTotal)
datFIN_sca<- cbind(datTotal_sca, probTotal)

# Check that there are people of each group in train and test
unique(datTrain$group)
unique(datTest$group)

saveRDS(datTrain, "train_clean_g")
saveRDS(datTest, "test_clean_g")
saveRDS(datTotal, "total_clean_g")
saveRDS(datTrain_sca, "train_clean_sca_g")
saveRDS(datTest_sca, "test_clean_sca_g")
saveRDS(datTotal_sca, "total_clean_sca_g")

## 7.1 Group accuracy ----
Acc_group<- data.frame(model = colnames(binTrain)[c(3:ncol(binTrain))])
for(i in c(1:5)){
    col<- paste("gro", i, sep="")
    in_g<- binTrain[binTrain$PassengerId %in% rownames(datTrain[datTrain$group == i, ]),]
    colum = c()
    for(j in seq(3,ncol(in_g))){
        acc<- 1 - (sum(abs(as.numeric(in_g[,j]) - as.numeric(in_g[,2])))/nrow(in_g))
        colum<- c(colum, acc)
    }
    Acc_group[[col]]<- colum
}

# 7.2 Predictions of the best ----
tail(Acc_group[order(Acc_group$gro1),], 5)
tail(Acc_group[order(Acc_group$gro2),], 5)
tail(Acc_group[order(Acc_group$gro3),], 5)
tail(Acc_group[order(Acc_group$gro4),], 5)
tail(Acc_group[order(Acc_group$gro5),], 5)

df_names<- c()
pred_mean<- factor()
pred_mean_pond<- factor()
for(i in c(1:5)){
    # Selection
    best<- as.character(tail(Acc_group[order(Acc_group$gro1),], 5)$model)
    acc<- as.numeric(tail(Acc_group[order(Acc_group$gro1),], 5)[[paste("gro", i, sep="")]])
    pon<- acc/sum(acc)
    g_names<- rownames(datTest[datTest$group == i, ])
    in_g<- probTest[probTest$PassengerId %in% g_names, best]
    # Evaluation
    m_pred<- ifelse(rowMeans(in_g)<0.5, 0,1)
    m_p_pred<- ifelse(rowSums(pon * in_g)<0.5, 0,1)
    df_names<- c(df_names, g_names)
    pred_mean<- c(pred_mean, m_pred)
    pred_mean_pond<- c(pred_mean_pond, m_p_pred)
}
Sub_best_mean<- data.frame(PassengerId = as.numeric(df_names), Survived = as.factor(pred_mean))
Sub_best_mean<- Sub_best_mean[order(Sub_best_mean$PassengerId),]
Sub_best_mean_pond<- data.frame(PassengerId = as.numeric(df_names), Survived = as.factor(pred_mean_pond))
Sub_best_mean_pond<- Sub_best_mean_pond[order(Sub_best_mean_pond$PassengerId),]

# Cuantos cambian su destino
all(as.numeric(Sub_best_mean$Survived) - as.numeric(Sub_best_mean_pond$Survived) == 0)

fwrite(Sub_best_mean, "./sub/best_mean_sub.csv", row.names = FALSE)
fwrite(Sub_best_mean_pond, "./sub/best_mean_pond_sub.csv", row.names = FALSE)
