library(plyr)
library(ggplot2)
library(reshape2)

# Import prediction results
x <- readRDS(file = "../models/ML.final.summary.rds") # from BLOCK04b
x <- subset(x, rank == "Genus" | rank == "ASV")
x <- subset(x, dataset == "minimal.norm")

###
### Compare genus vs. ASV

# ANOVA
fit <- lm(prediction.evaluation ~ factor*rank, subset(x, model == "L2LinearSVM" & mode == "regression"))
anova(fit)
fit <- lm(prediction.evaluation ~ factor*rank, subset(x, model == "RandomForest" & mode == "regression"))
anova(fit)

# How many instances 
foo <- reshape(x[,c("model","factor","rank","seed","prediction.evaluation")], direction = "wide", idvar = c("model","factor","seed"), timevar = "rank")
foo$diff <- foo$prediction.evaluation.ASV/foo$prediction.evaluation.Genus
nrow(subset(foo, diff < 1))/nrow(foo) # only models which fared poorly


###
### Plot Best Predictors

## Regression
plot_me <- subset(x, rank == "ASV" & mode == "regression" & dataset == "minimal.norm" & model == "RandomForest")

# remove minor elements and potassium ratings due to narrow dynamic ranges
plot_me <- subset(plot_me, factor != "minor_elements_rating" & factor != "K_rating")

# order by best average accuracy
order_me <- ddply(plot_me, ~ factor, summarise, avg = mean(prediction.evaluation))
order_me <- order_me[rev(order(order_me$avg)),]
plot_me$factor <- factor(plot_me$factor, levels = rev(order_me$factor))

# plot
plot <- ggplot(plot_me, aes(x=factor, y=prediction.evaluation, colour = factor)) + geom_boxplot(outlier.shape = NA) + ylab("R2") + facet_grid(~model)
plot <- plot + geom_jitter(width=0.1, alpha=0.5) + coord_flip()
print(plot)
ggsave(plot, file = 'RandomForest.regression.accuracy.plot.pdf', height=5, width=10)


## Classification (note: RF was not run)
plot_me <- subset(x, rank == "ASV" & mode == "classification" & dataset == "minimal.norm" & model == "L2LinearSVM")

# remove minor elements due to poor balance among classes
plot_me <- subset(plot_me, factor != "minor_elements.category")

# order by best average accuracy
order_me <- ddply(plot_me, ~ factor, summarise, avg = mean(prediction.evaluation))
order_me <- order_me[rev(order(order_me$avg)),]
plot_me$factor <- factor(plot_me$factor, levels = rev(order_me$factor))

# plot
plot <- ggplot(plot_me, aes(x=factor, y=prediction.evaluation, colour = factor)) + geom_boxplot(outlier.shape = NA) + ylab("Kappa") + facet_grid(~model)
plot <- plot + geom_jitter(width=0.1, alpha=0.5) + coord_flip()
plot
ggsave(plot, file = 'SVM.classification.accuracy.plot.pdf', height=5, width=10)