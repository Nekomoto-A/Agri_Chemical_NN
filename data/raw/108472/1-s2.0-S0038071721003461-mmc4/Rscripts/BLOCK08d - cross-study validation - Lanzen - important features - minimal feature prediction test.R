library(phyloseq)
library(limma)
library(plyr)
library(tidyverse)
library(caret)
library(spaa)
library(parallel)
source('get_study.R')

parse_file <- function(file){
  file <- gsub("all.results_","",file)
  run.name <- unlist(strsplit(file, "_"))[1]
  seed <- unlist(strsplit(file, "_"))[3]
  dataset <- unlist(strsplit(file, "_"))[4]
  set.seed(as.numeric(seed))
  mo <- unlist(strsplit(file, "_"))[2]
  factor <- unlist(strsplit(file, paste(dataset,"",sep="_")))[2]
  factor <- gsub(".rds","",factor)
  
  if (grepl("Phylum|Class|Order|Family|Genus", file)){
    feature_type <- "Taxonomy"
    rank = unlist(strsplit(file, "_"))[length(unlist(strsplit(file, "_")))]
    rank <- gsub(".rds","",rank)
    factor <- gsub(paste("_",rank,sep=""),"",factor)
    
  } else {
    feature_type <- "ASV"
    rank <- "ASV"
    factor <- gsub(".rds","",factor)
  }
  
  return(list(run.name, seed, dataset, mo, factor, rank, feature_type))
}

###
### Models from various health metrics all have similar predictive accuracy for Lanzen. Is this because they share the same important features?

## Import and filter Lanzen Important Feature Data
x.lanz <- readRDS(file = "../models/lanzen.feature.importance.final.rds")
x.lanz <- subset(x.lanz, rank == "ASV" & dataset == "minimal.norm")
x.lanz <- x.lanz[order(x.lanz$ratio),]
x.lanz <- subset(x.lanz, ratio < 0.998 & basemetric > 0.1)
x.lanz <- unique(x.lanz[,c("model","health.metric","names")])

## How many are shared among all metrics?
venn.lanz <- subset(x.lanz, model == "L2LinearSVM")

# Draw a Venn Diagram
venn_bin <- data.frame(OTUs = unique(as.character(venn.lanz$names)))

for (metric in unique(venn.lanz$health.metric)){
  venn_bin[,metric] <- venn_bin$OTUs %in% subset(venn.lanz, health.metric == metric)$names
}

venn_counts <- vennCounts(venn_bin[,c("P_rating","water_capacity_rating","DNA","ace_soil_protein_index_rating","avg.rating")])
pdf("figures/venn.important.features.lanzen.pdf", width=8,height=6)
vennDiagram(venn_counts, names = c("P_rating","water_capacity_rating","DNA","ace_soil_protein_index_rating","avg.rating"), cex = 1, counts.col = "red")  
dev.off()

# Identify which features are core
row.names(venn_bin) <- venn_bin$OTUs
venn_bin$OTUs <- NULL
venn_bin <- venn_bin*1
venn_bin <- venn_bin[,-grep("soil_texture",colnames(venn_bin))]
foo <- venn_bin[which(rowSums(venn_bin) == ncol(venn_bin)),]
top.metrics <- colnames(foo)
top.features <- row.names(foo)

saveRDS(top.features, file = "lanzen.shared.important.features.rds") # for BLOCK08e

#top.metrics <- c("P_rating","water_capacity_rating","DNA","ace_soil_protein_index_rating","avg.rating")
#foo <- venn_bin[,top]
#nrow(foo[which(rowSums(foo) == ncol(foo)),])
#top.features <- row.names(foo[which(rowSums(foo) == ncol(foo)),])

###
### Predict Pastureland Yield with ONLY the 43 taxa overlapping among all five models

## Capture all regression models
files <- list.files(path = "data/results/", pattern="*.rds")

## Subset to best models
bestmodels <- readRDS("../models/lanzen.correlations.compiled.rds")
bestmodels <- subset(bestmodels, lanzen.factor %in% c("yield.dry.wt"))
bestmodels <- subset(bestmodels, health.metric %in% top.metrics)
bestmodels <- subset(bestmodels, permuted == 0 & r > 0 & dataset == "minimal.norm" & rank == "ASV" & model == "L2LinearSVM")

run_list <- vector(mode = "character")
for (file in files){
  find_me <- parse_file(file)
  find_me <- subset(bestmodels, model == find_me[[4]] & seed == find_me[[2]] & dataset == find_me[[3]] & health.metric == find_me[[5]] & rank == find_me[[6]])
  if (nrow(find_me) > 0){
    run_list <- append(run_list, file)
  }
}

## Import microbiome data
p <-  readRDS(file = "../microbiome/p_lanzen.final.rds")

# Read count data
if(any(taxa_sums(p) == 0)){
  p <- subset_taxa(p, taxa_sums(p) > 0)
}

# Subset to key taxa
p <- subset_taxa(p, taxa_names(p) %in% top.features)

# Prepare metadata and select only 
meta <- as.data.frame(as(sample_data(p), "matrix"), stringsAsFactors = F) %>%
  select("yield.dry.wt")
meta$sample <- row.names(meta)

# make count table
counts <- as.data.frame(as(t(otu_table(p)), "matrix"))

# add in sample info
counts$sample <- row.names(counts)

# Merge counts and metadata
data <- inner_join(meta, counts, by=c("sample"="sample")) %>%
  select(-sample) %>%
  drop_na()
data <- data[,c(2:ncol(counts),1)]
data[,"yield.dry.wt"] <- as.numeric(data[,"yield.dry.wt"])

# Repeat transformation
preProcValues <- preProcess(data, method = "range")
main.data <- predict(preProcValues, data)

count <- 1
for (file in run_list){
  
  ## Import necessary data (model, validation data (i.e. testData) count table)
  info <- unlist(parse_file(file))
  
  metric <- info[5]
  run.name <- info[1]
  seed <- info[2]
  set.seed(as.numeric(seed))
  dataset <- info[3]
  mo <- info[4]
  rank <- info[6]
  feature_type <- info[7]
  
  # assign evaluation method
  met <- "Rsquared"
  
  # Import model
  model <- readRDS(paste("../models/regression/",file,sep=""))[4]
  
  # Consolidate model and testing feature sets (subset predictors in dataset to those present in model and vice versa)
  if (mo == "RandomForest"){
    features <- gsub("trainingData.","",colnames(as.data.frame(model[[1]][13])))
  } else {
    features <- unlist(predictors(model))
    features <- gsub("\`","", features)
  }
  
  # remove those in the test set absent from model
  dataTransformed <- cbind(main.data[,which(colnames(main.data) %in% features)],main.data[,ncol(main.data)])
  colnames(dataTransformed)[ncol(dataTransformed)] <- "yield.dry.wt"
  
  # add in dummy info for ASVs in the model, but absent from data
  foo <- data.frame(ASV = setdiff(features, taxa_names(p)), stringsAsFactors = F)
  row.names(foo) <- foo$ASV
  foo <- data.frame(t(foo), check.names = F, stringsAsFactors = F)
  foo[1:nrow(dataTransformed),] <- 0
  foo[] <- lapply(foo, function(x) as.numeric(as.character(x)))
  dataTransformed <- cbind(dataTransformed, foo)
  
  ## Run model
  predictions <- predict(model, dataTransformed)[[1]]
  base_metric <- summary(lm(predictions ~ dataTransformed[,"yield.dry.wt"]))$r.squared

  # Save
  if (count == 1){
    results <- data.frame(model = mo, seed = seed, dataset = dataset, rank = rank, health.metric = metric, lanzen.factor = "yield.dry.wt", R2 = base_metric, stringsAsFactors = F)
    results2 <- data.frame(model = mo, seed = seed, dataset = dataset, rank = rank, health.metric = metric, lanzen.factor = "yield.dry.wt", obs.data = dataTransformed[,"yield.dry.wt"], pred.data = predictions, stringsAsFactors = F)
    count <- count + 1
  } else {
    results <- rbind(results, data.frame(model = mo, seed = seed, dataset = dataset, rank = rank, health.metric = metric, lanzen.factor = "yield.dry.wt", R2 = base_metric, stringsAsFactors = F))
    results2 <- rbind(results2, data.frame(model = mo, seed = seed, dataset = dataset, rank = rank, health.metric = metric, lanzen.factor = "yield.dry.wt", obs.data = dataTransformed[,"yield.dry.wt"], pred.data = predictions, stringsAsFactors = F))
  }
}

saveRDS(results, file = "data/lanzen.minimum.feature.set.results.rds")
saveRDS(results2, file = "data/lanzen.minimum.feature.set.plot.data.rds")