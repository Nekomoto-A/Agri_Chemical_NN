library(phyloseq)
library(Hmisc)
library(limma)
source('get_study.R')

###
### Identify Overlap between *Important* Features in SH models and ALL Features in Aurora

# Import Aurora microbiome data
p <- get_study("minimal.norm", "aurora")   
p <- subset_taxa(p, taxa_sums(p) > 0)

# Total OTU overlap
v <- readRDS(file = "../microbiome.data/minimal.ASV.datset.rds") # from BLOCK03
length(intersect(taxa_names(p), taxa_names(v)))
length(taxa_names(v))

# Import Feature Importance data for SH models
x <- readRDS(file = "../models/compiled.feature.selection.results.final.rds") # output from "ML.block03.feature.importance.SH.R"
x <- subset(x, rank == "ASV")
x <- x[order(x$ratio),]

result <- data.frame(health.metric = NA, overlap = NA, stringsAsFactors = F)
for (metric in unique(x$health.metric)){
  foo <- subset(x, health.metric == metric & dataset == "minimal.norm" & model == "RandomForest" & ratio < 0.998)
  
  if (nrow(foo) > 0){
    olap <- length(taxa_names(subset_taxa(p, taxa_names(p) %in% foo$names)))
    result <- rbind(result, data.frame(health.metric = metric, overlap = olap, stringsAsFactors = F))
  } else {
    result <- rbind(result, data.frame(health.metric = metric, overlap = NA, stringsAsFactors = F))
  }
}

write.csv(result, file = "figures/total.overlap.aurora.csv", quote = F, row.names = F)


###
### Identify Overlap between *Important* Features in SH models and *Important* Features in Aurora

# Import and filter Aurora
x.aurora <- readRDS(file = "../models/aurora.feature.importance.final.rds")
x.aurora <- subset(x.aurora, rank == "ASV" & dataset == "minimal.norm")
x.aurora <- x.aurora[order(x.aurora$ratio),]
x.aurora <- subset(x.aurora, ratio < 0.998 & basemetric > 0.1)
x.aurora <- unique(x.aurora[,c("model","health.metric","names")])

# Import and filter SH
x.SH <- readRDS(file = "../models/compiled.feature.selection.results.final.rds") # output from "ML.block03.feature.importance.SH.R"
x.SH <- subset(x.SH, rank == "ASV" & dataset == "minimal.norm")
x.SH <- x.SH[-grep("category",x.SH$health.metric),]
x.SH <- x.SH[order(x.SH$ratio),]
x.SH <- subset(x.SH, ratio < 0.998)
x.SH <- unique(x.SH[,c("model","health.metric","names")])

results <- data.frame(model = NA, health.metric = NA, overlap = NA, total.aurora = NA, total.SH = NA, stringsAsFactors = F)
for (mod in c("L2LinearSVM","RandomForest")){
  for (metric in unique(x.aurora$health.metric)){
    foo <- subset(x.aurora, health.metric == metric & model == mod)
    foo.SH <- subset(x.SH, health.metric == metric & model == mod)
    
    if (nrow(foo) > 0){
      olap <- length(which(unique(foo$names) %in% foo.SH$names))
      results <- rbind(results, data.frame(model = mod, health.metric = metric, overlap = olap, total.aurora = length(unique(foo$names)), total.SH = length(unique(foo.SH$names)), stringsAsFactors = F))
    } else {
      results <- rbind(results, data.frame(model = mod, health.metric = metric, overlap = NA,  total.aurora = NA, total.SH = NA, stringsAsFactors = F))
    }
  }
}

results <- results[complete.cases(results),]
results$percent.aurora <- round((results$overlap/results$total.aurora)*100,1)
results$percent.SH <- round((results$overlap/results$total.SH)*100,1)
results$percent.metric <- round((results$overlap/(results$total.SH+results$total.aurora-results$overlap))*100,1)
results <- results[rev(order(results$model, results$percent.metric)),]
results
write.csv(results, file = "figures/overlap.important.features.aurora.csv", quote = F, row.names = F)

##
## Correlate accuracy with overlap

x <-  readRDS(file = "../microbiome.data/aurora.correlations.final.rds")  # from BLOCK07a
x <- subset(x, rank == "ASV" & dataset == "minimal.norm" & pearsons.r > 0 & test.set == "real")
x.accuracy <- ddply(x, ~ health.metric + model, summarise, avg.accuracy = mean(R.squared))

# merge data
y <- merge(results, x.accuracy, by = c("model","health.metric"))

# test correlation
rcorr(y$avg.accuracy, y$percent.metric)
rcorr(y$avg.accuracy, y$percent.aurora)
rcorr(y$avg.accuracy, y$percent.SH)

##
## Who are the overlapping features predictive of health rating?

## Venn diagram 
SH.SVM <- as.character(subset(x.SH, model == "L2LinearSVM" & health.metric == "avg.rating")$names)
SH.RF <- as.character(subset(x.SH, model == "RandomForest" & health.metric == "avg.rating")$names)
aurora.SVM <- as.character(subset(x.aurora, model == "L2LinearSVM" & health.metric == "avg.rating")$names)
aurora.RF <- as.character(subset(x.aurora, model == "RandomForest" & health.metric == "avg.rating")$names)

# Draw a Venn Diagram
venn_bin <- data.frame(OTUs = unique(c(SH.SVM, SH.RF, aurora.SVM, aurora.RF)))
venn_bin[,"SH.RF"] <- venn_bin$OTUs %in% SH.RF
venn_bin[,"SH.SVM"] <- venn_bin$OTUs %in% SH.SVM
venn_bin[,"Aurora.SVM"] <- venn_bin$OTUs %in% aurora.SVM
venn_bin[,"Aurora.RF"] <- venn_bin$OTUs %in% aurora.RF

venn_counts <- vennCounts(venn_bin[,c("SH.RF","SH.SVM","Aurora.SVM","Aurora.RF")])
pdf("figures/venn.important.features.models.pdf", width=8,height=6)
vennDiagram(venn_counts, names = c("SH.RF","SH.SVM","Aurora.SVM","Aurora.RF"), cex = 1, counts.col = "red")  
dev.off()

row.names(venn_bin) <- venn_bin$OTUs
venn_bin$OTUs <- NULL
venn_bin <- venn_bin*1
venn_bin[which(rowSums(venn_bin) == 4),]

foo <- subset(x.aurora, health.metric == "avg.rating" & model == "L2LinearSVM")
foo$rank <- 1:nrow(foo)
foo.SH <- subset(x.SH, health.metric == "avg.rating" & model == "L2LinearSVM")
foo.SH$rank <- 1:nrow(foo.SH)
olap <- as.character(unique(foo$names)[which(unique(foo$names) %in% foo.SH$names)])
subset(foo, names %in% olap)
subset(foo.SH, names %in% olap)


###
### Models from various health metrics all have similar predictive accuracy for Aurora

## Import and filter Lanzen Important Feature Data
x.aurora <- readRDS(file = "../models/aurora.feature.importance.final.rds")
x.aurora <- subset(x.aurora, rank == "ASV" & dataset == "minimal.norm")
x.aurora <- x.aurora[order(x.aurora$ratio),]
x.aurora <- subset(x.aurora, ratio < 0.998 & basemetric > 0.1)
x.aurora <- unique(x.aurora[,c("model","health.metric","names")])

## How many are shared among all metrics?
venn.aurora <- subset(x.aurora, model == "L2LinearSVM")

# Draw a Venn Diagram
venn_bin <- data.frame(OTUs = unique(as.character(venn.aurora$names)))

for (metric in unique(venn.aurora$health.metric)){
  venn_bin[,metric] <- venn_bin$OTUs %in% subset(venn.aurora, health.metric == metric)$names
}

venn_bin <- venn_bin[,-grep("ace",colnames(venn_bin))]
rownames(venn_bin) <- venn_bin$OTUs
venn_bin$OTUs <- NULL

venn_counts <- vennCounts(venn_bin)
pdf("figures/venn.important.features.aurora.pdf", width=8,height=6)
vennDiagram(venn_counts, names = colnames(venn_bin), cex = 1, counts.col = "red")  
dev.off()