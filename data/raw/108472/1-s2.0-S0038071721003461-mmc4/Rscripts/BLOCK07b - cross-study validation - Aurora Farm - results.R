library(ggplot2)
library(reshape2)
library(plyr)
source('get_study.R')

# Import results
x <-  readRDS(file = "../microbiome.data/aurora.correlations.final.rds")  # from BLOCK07a
x <- subset(x, rank == "ASV" & dataset == "minimal.norm" & pearsons.r > 0)

##
## Compare to permuted data
mod <- "RandomForest"
sig <- subset(x,  model == mod)

# Calculate False Discovery Rate
sig$sig <- 0
sig$sig[which(sig$p.value < 0.05)] <- 1
table(sig$test.set, sig$sig)

#  FDR = 0/201  # Random forest
#  FDR = 0/278    # SVM

##
## Plot accuracy for predicting Aurora health ratings

# filter to models of interest
x <- subset(x, test.set == "real") # SVM performed better

# remove minor elements and potassium ratings due to narrow dynamic ranges
plot_me <- subset(x, health.metric != "minor_elements_rating" & health.metric != "K_rating")

# order by best average accuracy
order_me <- ddply(plot_me, ~ health.metric, summarise, avg = mean(R.squared))
order_me <- order_me[rev(order(order_me$avg)),]
plot_me$health.metric <- factor(plot_me$health.metric, levels = rev(order_me$health.metric))

# plot
plot_me$model <- factor(plot_me$model, levels = c("RandomForest","L2LinearSVM"))
plot <- ggplot(plot_me, aes(x=health.metric, y=R.squared, color = model)) + geom_boxplot(outlier.shape = NA) + ylab("R2") + ggtitle(mod) 
plot <- plot + geom_point(position=position_jitterdodge(jitter.width = 0.1), alpha=0.5) + coord_flip()
plot
ggsave(plot, file = 'figures/Aurora.regression.accuracy.plot.pdf', height=5, width=10)


##
## Predictions are correlated, but fall within a much narrower range. Well off the identity (1:1) line.

# Import all prediction / observed data points
x <- readRDS(file = "data/aurora.predictions.final.rds")
x <- subset(x, test.set == "real")

#saveRDS(unique(x$health.metric), file = "data/all.SH.rating.names.rds")

# Untransform data 
p <- get_study("minimal.norm", "aurora")   
obs <- as.data.frame(as(sample_data(p), "matrix"), stringsAsFactors = F)
sampleIDs <- row.names(obs)
obs <- obs[,unique(x$health.metric)]
obs <- as.data.frame(lapply(obs, function(x) as.numeric(x)))
obs$sampleID <- sampleIDs
obs <- melt(obs)

# cycle through each metric; regress observed and normalized data
count <- 1
for (metric in unique(x$health.metric)){
  normed <- unique(subset(x, health.metric == metric)[,c("aurora.sampleID","observed")])
  norm_me <- merge(normed, subset(obs, variable == metric), by.x = "aurora.sampleID", by.y = "sampleID")
  
  #plot(norm_me$value, norm_me$observed)
  m <- coefficients(lm(value ~ observed, data = norm_me))[2]
  b <- coefficients(lm(value ~ observed, data = norm_me))[1]
  
  if (count == 1){
    coeffs <- data.frame(health.metric = metric, m = m, b = b, stringsAsFactors = F)
    count <- count + 1
  } else {
    coeffs <- rbind(coeffs, data.frame(health.metric = metric, m = m, b = b, stringsAsFactors = F))
  }
}

# Make dataframe of all models
models <- unique(x[,c("model","seed","dataset","health.metric","rank")])
models$predicted.range <- NA
models$obs.range <- NA

# run through all models and calculate the ratio of prediction range to observed range
for (n in 1:nrow(models)){
  foo <- subset(x, model == models$model[n] & seed == models$seed[n] & dataset == models$dataset[n] & health.metric == models$health.metric[n] & rank == models$rank[n])
  
  # denormalize data
  m <- coeffs[grep(models$health.metric[n],coeffs$health.metric),"m"]
  b <- coeffs[grep(models$health.metric[n],coeffs$health.metric),"b"]
  foo$obs <- foo$observed*m + b
  foo$pred <- foo$predicted*m + b

  models$obs.range[n] <- max(foo$obs)-min(foo$obs)
  models$pred.range[n] <- max(foo$pred)-min(foo$pred)
}

foobie1 <- ddply(subset(models, dataset == "minimal.norm" & rank == "ASV"), ~ model + health.metric, summarise, avg.pred.range = mean(pred.range))
foobie2 <- ddply(subset(models, dataset == "minimal.norm" & rank == "ASV"), ~ model + health.metric, summarise, avg.obs.range = mean(obs.range))
foobie <- merge(foobie1, foobie2, by = c("model","health.metric"))
foobie <- foobie[rev(order(foobie$model, foobie$avg.pred.range)),]
foobie$avg.pred.range <- round(foobie$avg.pred.range, 2)
foobie$avg.obs.range <- round(foobie$avg.obs.range, 2)

write.csv(reshape(foobie, direction = "wide", idvar = "health.metric", timevar =  "model"), file = "figures/Aurora.prediction.dynamic.range.csv", quote = F, row.names = F)

##
## Is there correlation between dynamic range and prediction accuracy?

# calculate average accuracy
x <-  readRDS(file = "data/aurora.correlations.final.rds")  # from BLOCK07a
x <- subset(x, rank == "ASV" & dataset == "minimal.norm" & pearsons.r > 0 & test.set == "real" & model == "L2LinearSVM")
x.accuracy <- ddply(x, ~ health.metric, summarise, avg.accuracy = mean(R.squared))

# merge data
y <- merge(subset(foobie, model == "L2LinearSVM"), x.accuracy, by = "health.metric")  
y$prop <- y$avg.pred.range/y$avg.obs.range

# test correlation
rcorr(y$avg.accuracy, y$avg.pred.range)
rcorr(y$avg.accuracy, y$avg.obs.range)
rcorr(y$avg.accuracy, y$prop)

plot(x$avg.accuracy ~ x$avg.dynamic)


##
## What is the accuracy for predicting tillage in Aurora?
x <- rbind(readRDS(file = "data/aurora.classification.predictions.Genus.final.rds"),  readRDS(file = "data/aurora.classification.predictions.ASV.final.rds"))
foo <- ddply(x, ~ rank + test.set + model, summarise, avg.kappa = mean(kappa))
foo$avg.kappa <- round(foo$avg.kappa, 3)

