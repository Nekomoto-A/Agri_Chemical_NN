library(plyr)
library(reshape2)
library(Hmisc)
library(ggplot2)

# Import all prediction / observed data points
x <- readRDS(file = "../models/SH.predictions.eval.for.plotting.rds")

# Make dataframe of all models
models <- unique(x[,c("model","seed","dataset","factor","rank")])
models$predicted.range <- NA
models$obs.range <- NA

# quick plot of health rating
#foo <- subset(x, model == "RandomForest" & dataset == "minimal.norm" & factor == "avg.rating" & rank == "ASV")
#foo <- merge(ddply(foo, ~ sampleID, summarise, avg.obs = mean(obs.data)), ddply(foo, ~ sampleID, summarise, avg.pred = mean(predicted.data)), by = "sampleID")
#rcorr(foo$avg.obs, foo$avg.pred)
#foo <- ggplot(foo, aes(avg.obs, avg.pred)) + geom_point() + geom_smooth(method = "lm", se = FALSE)
#ggsave(foo + theme(legend.position="none"), filename='foo.pdf', height=10, width=16)

# run through all models and calculate the ratio of prediction range to observed range
for (n in 1:nrow(models)){
  foo <- subset(x, model == models$model[n] & seed == models$seed[n] & dataset == models$dataset[n] & factor == models$factor[n] & rank == models$rank[n])
  models$obs.range[n] <- max(foo$obs.data)-min(foo$obs.data)
  models$predicted.range[n] <- max(foo$predicted.data)-min(foo$predicted.data)/(max(foo$obs.data)-min(foo$obs.data))
  
  #ggplot(foo, aes(x=predicted.data, y=obs.data)) + geom_point() +  geom_smooth(method=lm, se=FALSE)
}

foobie1 <- ddply(subset(models, dataset == "minimal.norm" & rank == "ASV"), ~ model + factor, summarise, avg.predicted = mean(predicted.range))
foobie2 <- ddply(subset(models, dataset == "minimal.norm" & rank == "ASV"), ~ model + factor, summarise, avg.obs = mean(obs.range))
foobie <- merge(foobie1, foobie2, by = c("model","factor"))
foobie <- foobie[rev(order(foobie$model, foobie$avg.predicted)),]
foobie$avg.predicted <- round(foobie$avg.predicted, 2)
foobie$avg.obs <- round(foobie$avg.obs, 2)

write.csv(foobie, file = "figures/SH.prediction.dynamic.range.csv", quote = F, row.names = F)
saveRDS(foobie, file = "../models/SH.prediction.dynamic.range.rds")


##
## Is there correlation between dynamic range and prediction accuracy?

# import dynamic range data
x.range <- readRDS(file = "../models/SH.prediction.dynamic.range.rds")

# calculate average accuracy
x <- readRDS(file = "../models/ML.final.summary.rds") # from BLOCK04b
x.accuracy <- ddply(subset(x, rank == "ASV" & mode == "regression" & dataset == "minimal.norm"), ~ factor + model, summarise, avg.accuracy = mean(prediction.evaluation))

# merge data
x <- merge(x.range, x.accuracy, by = c("model","factor"))  

# test correlation
rcorr(x$avg.accuracy, x$avg.obs)
rcorr(x$avg.accuracy, x$avg.predicted)

plot(x$avg.accuracy ~ x$avg.dynamic)