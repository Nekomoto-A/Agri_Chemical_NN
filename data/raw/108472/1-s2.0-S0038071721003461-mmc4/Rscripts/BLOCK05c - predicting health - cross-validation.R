library(plyr)
library(ggplot2)
library(Hmisc)

# Import data
x <- readRDS(file = "../models/ML.final.summary.rds")

# Absolute difference between final model and best model from cross-validation step (based on an subset of the training set)
x$diff <- x$prediction.evaluation - x$cross.validation

##
## Plot difference across normalization methods, models and rnaks
x$dataset <- factor(x$dataset, levels = c("minimal.norm","filter.norm","filter.css","minimal.rare","filter.rare"))

# Plot Classification
plot <- ggplot(subset(x, mode == "classification" & factor == "health.category"), aes(x=rank, y=diff, colour = dataset)) + geom_boxplot(outlier.shape = NA) + ylab("Kappa") + facet_grid(~model)
plot <- plot + geom_point(position=position_jitterdodge(jitter.width = 0.05), alpha=0.5)
plot

fit <- lm(diff ~  model*dataset*rank, subset(x,factor == "health.category"))
anova(fit)

# Plot Regression
plot <- ggplot(subset(x, mode == "regression" & factor == "avg.rating"), aes(x=rank, y=diff, colour = dataset)) + geom_boxplot(outlier.shape = NA) + ylab("Rsquared") + facet_grid(~model)
plot <- plot + geom_point(position=position_jitterdodge(jitter.width = 0.05), alpha=0.5)
plot

fit <- lm(diff ~ model*dataset*rank, subset(x, factor == "avg.rating"))
anova(fit)

ddply(x, ~ model + rank, summarise, avg.diff = mean(diff))


##
## By factor

for (mo in c("L2LinearSVM", "RandomForest")){
  foo <- subset(x, rank == "ASV" & dataset == "minimal.norm" & mode == "regression" & model == mo)
  fit <- lm(diff ~ factor, foo)
  foo <- as.data.frame(TukeyHSD(aov(fit))$factor)
  colnames(foo)[4]<-"p.adj"
  print(subset(foo, p.adj < 0.05))  
}
