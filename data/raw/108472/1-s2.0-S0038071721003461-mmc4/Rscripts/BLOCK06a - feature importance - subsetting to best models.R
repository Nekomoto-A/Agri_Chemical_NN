library(plyr)

## This information is used in "ML.block03.feature.importance.SH.R"

###########################################
## Select Best Models for Feature Selection

# Import model evaluation data (from BLOCK04b)
x <- readRDS(file = "../models/ML.final.summary.rds")

## Remove poor performing models

# SVM performed better than RF at classification. Keep Family, Genus, ASV for SVM and ASV for RF.
x.class <- subset(x, mode == "classification" & rank %in% c("Family","Genus","ASV"))
x.class <- x.class[-which(x.class$model == "RandomForest" & x.class$rank %in% c("Family","Genus")),]

# sanity check
#ggplot(subset(x.class, factor == "health.category"), aes(x=rank, y=prediction.evaluation, colour = dataset)) + geom_boxplot(outlier.shape = NA) + ylab("Kappa") + facet_grid(~model)  + geom_point(position=position_jitterdodge(jitter.width = 0.05), alpha=0.5)

# RF performed better than SVM at regression. Do the reverse of SVM
x.regress <- subset(x, mode == "regression" & rank %in% c("Family","Genus","ASV"))
x.regress <- x.regress[-which(x.regress$model == "L2LinearSVM" & x.regress$rank %in% c("Family","Genus")),]

# sanity check
#ggplot(subset(x.regress, factor == "avg.rating"), aes(x=rank, y=prediction.evaluation, colour = dataset)) + geom_boxplot(outlier.shape = NA) + ylab("Rsquared") + facet_grid(~model)  + geom_point(position=position_jitterdodge(jitter.width = 0.05), alpha=0.5)

# Remove all models based on rarefied data
y <- rbind(x.class, x.regress)
y <- subset(y, dataset != "minimal.rare" & dataset != "filter.rare")


####
#### Identify best regression models

# Subset by significance of Pearsons' correlation
bestR <- subset(y, p.value < 0.01)

# Order
bestR <- bestR[rev(order(bestR$model, bestR$feature, bestR$factor, bestR$prediction.evaluation)),]

#Subset top 10 for each category
count <- 1
fact <- unique(bestR$factor)[1]
for (mo in unique(bestR$model)){
  for (feat in unique(bestR$rank)){
    for (fact in unique(bestR$factor)){
      
      # select best model for each seed
      best <- subset(bestR, model == mo & feature == feat & factor == fact)
      
      if (any(duplicated(best$seed))){
        best <- best[-which(duplicated(best$seed)),]
      }
      
      if (count == 1){
        if (nrow(best) > 10){
          keepR <- best[1:10,]
        } else {
          keepR <- best
        }
        count <- count + 1
        
      } else {
        if (nrow(best) > 10){
          keepR <- rbind(keepR, best[1:10,])
        } else {
          keepR <- rbind(keepR, best)      
        }
      }
    }
  }
}
keepR$product <- NULL

#hist(subset(x, mode == "classification")$prediction.evaluation)
#hist(subset(x, mode == "regression")$prediction.evaluation)
#hist(subset(x, mode == "regression")$pearsons.r)

####
#### Identify best classification models

bestC <- subset(y, evaluation.metric == "Kappa")

# Subset top 10 for each category
bestC <- bestC[rev(order(bestC$model, bestC$feature, bestC$factor, bestC$prediction.evaluation)),]

count <- 1
for (mo in unique(bestC$model)){
  for (feat in unique(bestC$feature)){
    for (fact in unique(bestC$factor)){
      
      # select best model for each seed
      best <- subset(bestC, model == mo & feature == feat & factor == fact)
      
      if (any(duplicated(best$seed))){
        best <- best[-which(duplicated(best$seed)),]
      }
      
      if (count == 1){
        if (nrow(best) > 10){
          keepC <- best[1:10,]
        } else {
          keepC <- best
        }
        count <- count + 1
        
      } else {
        if (nrow(best) > 10){
          keepC <- rbind(keepC, best[1:10,])
        } else {
          keepC <- rbind(keepC, best)      
        }
      }
    }
  }
}

bestmodels <- rbind(keepR, keepC)
saveRDS(bestmodels, file = "data/bestmodels.SH.rds")

# sanity check
#ggplot(subset(bestmodels, factor == "health.category"), aes(x=rank, y=prediction.evaluation, colour = dataset)) + geom_boxplot(outlier.shape = NA) + ylab("Kappa") + facet_grid(~model)  + geom_point(position=position_jitterdodge(jitter.width = 0.05), alpha=0.5)
#ggplot(subset(bestmodels, factor == "avg.rating"), aes(x=rank, y=prediction.evaluation, colour = dataset)) + geom_boxplot(outlier.shape = NA) + ylab("Rsquared") + facet_grid(~model)  + geom_point(position=position_jitterdodge(jitter.width = 0.05), alpha=0.5)