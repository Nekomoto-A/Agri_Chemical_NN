library(plyr)
library(ggplot2)
library(reshape2)

# Import model evaluations
x <- read.table(file = "../models/ML.final.summary.results.tsv", sep="\t", header=T, stringsAsFactors = F)
x$factor <- gsub("_$","",x$factor)
x$rank <- factor(x$rank, levels = c("Order","Family","Genus","ASV"))
x$experiment <- NULL
saveRDS(x, file = "../models/ML.final.summary.rds")

###
### Figure 1A - Comparison of Processing Methods - Classification
x <- readRDS(file = "../models/ML.final.summary.rds")
x$dataset <- factor(x$dataset, levels = c("minimal.norm","filter.norm","filter.css","minimal.rare","filter.rare"))

# Plot Kappa
plot <- ggplot(subset(x, mode == "classification" & factor == "health.category"), aes(x=rank, y=prediction.evaluation, colour = dataset)) + geom_boxplot(outlier.shape = NA) + ylab("Kappa") + facet_grid(~model)
plot <- plot + geom_point(position=position_jitterdodge(jitter.width = 0.05), alpha=0.5)
plot
ggsave(plot, filename='figures/health.category.predictions.pdf', height=6, width=10)


###
### Figure 1B - Comparison of Processing Methods - Regression

# Plot Rsquared
plot <- ggplot(subset(x, mode == "regression" & factor == "avg.rating"), aes(x=rank, y=prediction.evaluation, colour = dataset)) + geom_boxplot(outlier.shape = NA) + ylab("Rsquared") + facet_grid(~model)
plot <- plot + geom_point(position=position_jitterdodge(jitter.width = 0.05), alpha=0.5)
plot
ggsave(plot, filename='figures/avg.health.rating.Rsquared.pdf', height=6, width=10)


###
### Table 1 - Gross Averages Regression
tableau <- ddply(subset(x, mode == "regression" & factor == "avg.rating" & model == "L2LinearSVM"), ~ dataset + rank, summarise, avg.pred = mean(prediction.evaluation))
tableau <- reshape(tableau, direction = "wide", idvar = "dataset", timevar =  "rank")
write.csv(tableau, file = "figures/table1.prediction.averages.SVM.csv", quote = F, row.names = F)


###
### Do Stats on Trends in normalization, sparsity filtering and aggregation

# Import data
x <- readRDS(file = "../models/ML.final.summary.rds")

## Add in factor groups (based on dataset names)
x$filtered <- 0
x$filtered[grep("filter", x$dataset)] <- 1

x$rarefied <- 0
x$rarefied[grep("rare", x$dataset)] <- 1


###
### Effect of rarefication and sparsity filtering

## classification

# SVM
fit <- lm(prediction.evaluation ~ rarefied * rank * filtered, subset(x, model == "L2LinearSVM" & factor == "health.category"))
anova(fit)

# RF
fit <- lm(prediction.evaluation ~ rarefied * rank * filtered, subset(x, model == "RandomForest" & factor == "health.category"))
anova(fit)

## regression

# SVM
fit <- lm(prediction.evaluation ~ rarefied * rank * filtered, subset(x, model == "L2LinearSVM" & factor == "avg.rating"))
anova(fit)

# RF
fit <- lm(prediction.evaluation ~ rarefied * rank * filtered, subset(x, model == "RandomForest" & factor == "avg.rating"))
anova(fit)


# Rarefying had significant effects
# Sparsity filtering has no significant impact
# There is a significant interaction between the two.


###
### Effect of model, proportioning (CSS vs. 1/read depth) and rank

## SVM

# regression
fit <- lm(prediction.evaluation ~ model*dataset*rank, subset(x, factor == "avg.rating" & rarefied == 0))
anova(fit)

# classification
fit <- lm(prediction.evaluation ~  model*dataset*rank, subset(x,factor == "health.category" & rarefied == 0))
anova(fit)

# 'dataset' is not significant, when subset to normalized datasets


##
## Model Variability

# Import data
x <- subset(readRDS(file = "../models/ML.final.summary.rds"), factor == "avg.rating" | factor == "health.category")

# Subset to best datasets
x <- subset(x, dataset == "filter.css" | dataset == "filter.norm" | dataset == "minimal.norm")
x <- subset(x, rank == "ASV")

ddply(x, ~ model + factor, summarise, sd = sd(prediction.evaluation))
