library(ggplot2)
library(reshape2)
library(plyr)
library(Hmisc)

# Import results
x <- readRDS("../models/lanzen.correlations.compiled.rds")
x <- subset(x, rank == "ASV" & dataset == "minimal.norm" & r > 0)

##
## Compare to permuted data
mod <- "L2LinearSVM"  # "RandomForest" | "L2LinearSVM"
sig <- subset(x,  model == mod)

# Calculate False Discovery Rate
sig$sig <- 0
sig$sig[which(sig$p < 0.05)] <- 1
table(sig$permuted, sig$sig)

#  FDR = 89/1510  # Random forest
#  FDR = 0/2975    # SVM

##
## Which Lanzen factor was best predicted?
x <- subset(x, permuted == 0)
x <- subset(x, health.metric != "minor_elements_rating" & health.metric != "K_rating")

# Plot Rsquared
plot <- ggplot(x, aes(x=model, y=R2, colour = health.metric)) + geom_boxplot(outlier.shape = NA) + ylab("Rsquared") + facet_wrap(~lanzen.factor)
plot <- plot + geom_point(position=position_jitterdodge(jitter.width = 0.05), alpha=0.5)
plot

# Output as Table S10
table10 <- ddply(subset(x, p < 0.05), ~ lanzen.factor + health.metric, summarise, avg.R2 = mean(R2))
table10 <- table10[-grep("soil_texture",table10$health.metric),]
table10$avg.R2 <- round(table10$avg.R2, 3)
table10 <- reshape(table10, direction = "wide", idvar = "lanzen.factor", timevar = "health.metric")
write.csv(table10, file = 'figures/pastureland.prediction.accuracy.csv', row.names = F, quote = F)

# reasonable accuracy for pH & yield
# low accuracy for CO2, OM and potassium 
# no accuracy for penetrability, compaction, induced.respiration 

###
### Figure 5C - Overall Accuracy

# reasonable accuracy for pH > yield > CO2/OM
factor <- "yield.dry.wt"  # "yield.dry.wt" | "pH" 

## add in minimal set experiment results  (FROM BLOCK08d)
if (factor == "yield.dry.wt"){
  y <- readRDS(file = ".../models/lanzen.minimum.feature.set.results.rds")
  y$model <- "L2LinearSVM.minimumset"  
  x <- rbind(x[,colnames(y)], y)
}

# Plot accuracy for yield
order_me <- ddply(subset(x, lanzen.factor == factor & model == "L2LinearSVM.minimumset"), ~ health.metric, summarise, avg = mean(R2))
order_me <- order_me[rev(order(order_me$avg)),]
plot_me <- subset(x, lanzen.factor == factor)
plot_me <- plot_me[-grep("soil_texture",plot_me$health.metric),]
plot_me$health.metric <-factor(plot_me$health.metric, levels = rev(order_me$health.metric))
plot_me$model <- factor(plot_me$model, levels = c("RandomForest","L2LinearSVM","L2LinearSVM.minimumset"))

plot <- ggplot(plot_me, aes(x=health.metric, y=R2, color = model)) + geom_boxplot(outlier.shape = NA) + ylab("Rsquared") + ggtitle(factor)
plot <- plot + geom_point(position=position_jitterdodge(jitter.width = 0.1), alpha=0.5) + coord_flip()
plot
ggsave(plot, filename='figures/Lanzen.yield.accuracy.plot.pdf', height=5, width=10)


###
### Figure 5D - Scatter plots of predicted vs. observed

y <- readRDS(file = "../models/lanzen.minimum.feature.set.plot.data.rds")
y$sampleID <- rep(1:198,length(unique(y$health.metric))*length(unique(y$seed)))
pred <- ddply(y, ~ sampleID + health.metric, summarise, mean.pred = mean(pred.data))
y <- merge(pred, unique(y[,c("sampleID","obs.data")]), by = "sampleID")
y <- subset(y, health.metric == "avg.rating")

# denormalize
p <-  readRDS(file = "../microbiome.data/p_lanzen.final.rds")
obs <- as.data.frame(as(sample_data(p), "matrix"), stringsAsFactors = F)
sampleIDs <- row.names(obs)
obs <- as.data.frame(lapply(obs, function(x) as.numeric(x)))
obs$sampleID <- sampleIDs
obs <- melt(obs)

# regress observed and normalized data
normed <- subset(y, health.metric = "avg.rating")[,c("sampleID","obs.data")]
norm_me <- cbind(normed, subset(obs, variable == "yield.dry.wt"))

#plot(norm_me$value, norm_me$observed)
m <- coefficients(lm(value ~ obs.data, data = norm_me))[2]
b <- coefficients(lm(value ~ obs.data, data = norm_me))[1]
plot(value ~ obs.data, data = norm_me)  

y$mean.pred <- y$mean.pred*m + b
y$obs.data <- y$obs.data*m + b

## Calculate regression coefficients and plot
rcorr(y$mean.pred, y$obs.data)
summary(lm(y$mean.pred ~ y$obs.data))  # R2 = 0.43; p < 0.001

plot <- ggplot(y, aes(obs.data, mean.pred)) + geom_point() + geom_smooth(method = "lm", se = FALSE)
plot
ggsave(plot, filename='figures/Lanzen.health.rating.model.by.yield.regression.pdf', height=5, width=10)
