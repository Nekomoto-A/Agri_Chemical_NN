library(plyr)
library(reshape2)
library(caret)
source("get_study.R")

# Import all prediction / observed data points
x <- readRDS(file = "../models/SH.predictions.eval.for.plotting.rds")
x <- subset(x, dataset == "minimal.norm" & rank == "ASV" & model == "L2LinearSVM")  # has to be SVM b/c classification not run for RF

# Determine the function to 'untransform' data
#meta <- readRDS(file = "../models/metadata.transformed")
#avg.rating <- meta$avg.rating
#preProcValues <- preProcess(meta, method = "range")
#dataTransformed <- predict(preProcValues, meta)
#comp <- data.frame(original = avg.rating, transformed = dataTransformed$avg.rating)
#plot(comp$original~comp$transformed)
#fit <- lm(comp$original~comp$transformed)
# original =  trnasformed*73.43 + 21.64

# Convert predicted ratings to categories
x$converted.pred <- x$predicted.data*73.43 + 21.64
x$pred.cat <- cut(x$converted.pred, c(0,20,40,60,80,100))
x$pred.cat <- factor(x$pred.cat, levels = c("(0,20]","(20,40]","(40,60]","(60,80]","(80,100]"))

# Do same for observation
x$converted.obs <- x$obs.data*73.43 + 21.64
x$obs.cat <- cut(x$converted.obs, c(0,20,40,60,80,100))
x$obs.cat <- factor(x$obs.cat, levels = c("(0,20]","(20,40]","(40,60]","(60,80]","(80,100]"))

# sanity check
#p <- get_study("minimal.norm", "full")
#p <- prune_samples(readRDS(file = "../microbiome.data/ML.sample.set.rds"), p)
#meta <- as.data.frame(as(sample_data(p), "matrix"), stringsAsFactors = F)
#meta$sampleID <- row.names(meta)
#foo <- unique(subset(x, factor == "avg.rating")[,c("sampleID","obs.data","converted.obs","obs.cat")])
#foo <- foo[order(foo$obs.data),]
#meta <- meta[order(meta$avg.rating),]
#foo$original <- meta$avg.rating
#foo$originalID <- row.names(meta)
# sampleID are all mixed up.
#foo <- merge(meta[,c("sampleID","health.category","avg.rating")],unique(subset(x, factor == "avg.rating")[,c("sampleID","obs.data","converted.obs","obs.cat")]))


##
## Calculate Kappa from rating predictions
x$comboID <- paste(x$model, x$seed, x$factor,sep=";")

count <- 1
for (id in unique(x$comboID)){
  foo <- subset(x, comboID == id)
  foo <- confusionMatrix(data = as.factor(foo$pred.cat), reference = as.factor(foo$obs.cat))
  confusion <- as.data.frame(foo$table)
  confusion$comboID <- id
  k <- foo$overall[2]

  if (count == 1){
    results <- data.frame(comboID = id, regress.kappa = k, stringsAsFactors = F)
    result2 <- confusion
    count <- count + 1    
  } else {
    results <- rbind(results, data.frame(comboID = id, regress.kappa = k, stringsAsFactors = F))
    result2 <- rbind(result2, confusion)
    
  }
}

# recover ID info
for (data in c("results", "result2")){
  foo <- get(data)
  foo$model <- unlist(lapply(foo$comboID, function(x) unlist(strsplit(x, ";"))[1]))
  foo$seed <- unlist(lapply(foo$comboID, function(x) unlist(strsplit(x, ";"))[2]))
  foo$factor <- unlist(lapply(foo$comboID, function(x) unlist(strsplit(x, ";"))[3]))
  foo$comboID <- NULL
  foo$factor <- gsub("avg.rating","health.category",foo$factor)
  foo$factor <- gsub("_rating",".category",foo$factor)
  assign(data, foo)  
}

saveRDS(results, file = "../models/posthoc.category.kappa.results.rds")
saveRDS(result2, file = "../models/posthoc.category.confusion.matrices.rds")


##
## Compare kappa from classification vs. regression
results <- readRDS(file = "../models/posthoc.category.kappa.results.rds")

y <- readRDS(file = "../models/ML.final.summary.rds") # from BLOCK04b
y <- subset(y, dataset == "minimal.norm" & rank == "ASV" & mode == "classification" & model == "L2LinearSVM")
colnames(y)[grep("prediction.evaluation", colnames(y))] <- "class.kappa"

# Merge and compare
z <- merge(results, y[,c("model","seed","factor","class.kappa")], by = c("model","seed","factor"))
z <- melt(z)
z$factor <- gsub(".category","",z$factor)
z <- subset(z, factor != "minor_elements")
z$factor <- factor(z$factor, levels = c("health", "ace_soil_protein_index", "active_carbon", "organic_matter", "respiration", "pH", "P", "aggregate_stability","water_capacity","surface_hardness","subsurface_hardness"))

plot <- ggplot(z, aes(x=value, color = variable)) + geom_histogram() + facet_wrap(~factor) + theme_bw() #+  xlim(-5, 5)
plot
ggsave(plot, filename='classification.vs.regression.histograms.pdf', height=10, width=16)

foo <- ddply(z, ~ factor + variable, summarise, avg.kappa = mean(value))
foo <- reshape(foo, direction = "wide", idvar = "factor", timevar = "variable")
mean(foo$avg.kappa.class.kappa- foo$avg.kappa.regress.kappa)


##
## Compare kappa from classification vs. regression

x <- readRDS(file = "../models/posthoc.category.confusion.matrices.rds")
x <- subset(x, model == "L2LinearSVM" & factor == "health.category")
y <- readRDS(file = "../models/actual.category.confusion.matrices.rds")

for (factor in c("Prediction","Reference")){
  x[,factor] <- gsub("\\(0,20\\]","cat0", x[,factor])
  x[,factor] <- gsub("\\(20,40\\]","cat1", x[,factor])
  x[,factor] <- gsub("\\(40,60\\]","cat2", x[,factor])
  x[,factor] <- gsub("\\(60,80\\]","cat3", x[,factor])
  x[,factor] <- gsub("\\(80,100\\]","cat4", x[,factor])
}

x <- subset(x, Prediction != "cat0" & Reference != "cat0")

x <- ddply(x, ~ Prediction + Reference, summarise, average.posthoc = mean(Freq))
y <- ddply(y, ~ Prediction + Reference, summarise, average.actual = mean(Freq))
final <- merge(x, y, by = c("Prediction","Reference"))
write.csv(final, file = "prediction.range.confusion.matrix.csv", row.names = F, quote = F)
