library(phyloseq)
library(plyr)
library(Hmisc)
library(tidyverse)
library(caret)
source('get_study.R')
source('get_melt.R')
source('down_sample_by_geogroup.R')

###
### Capture all regression models
files <- list.files(path = "../models/regression/", pattern="*.rds")

# target tillage
files <- files[grepl("tillage", files)]

# not interested in anything but Genus and ASV
files <- files[!grepl("Order|Family", files)]

count <- 1
for (permute_labels in c("permuted","real")){
  for (file in files){
    
    # Aurora data lacks information on the following measures:
    print(file)
    seed <- unlist(strsplit(file, "_"))[4]
    dataset <- unlist(strsplit(file, "_"))[5]
    set.seed(as.numeric(seed))
    mo <- unlist(strsplit(file, "_"))[3]
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
    
    # prep microbiome data
    p <- get_study(dataset, "aurora")   
    
    # Prepare metadata and select only sample Id and diagnosis columns
    meta <- as.data.frame(as(sample_data(p), "matrix"), stringsAsFactors = F) %>%
      select(factor)
    meta$sample <- row.names(meta)
    
    if (permute_labels == "permuted"){
      set.seed(70)
      meta[,factor] <- sample(meta[,factor])
    }
    
    if (feature_type == "ASV"){
      # Get rid of ASV with zero counts
      p <- subset_taxa(p, taxa_sums(p) > 0)
      
      # Read in OTU table and remove label and numOtus columns
      counts <- as.data.frame(as(t(otu_table(p)), "matrix"))
      
      # add in sample info
      counts$sample <- row.names(counts)
      
    } else {
      
      # Read in pre-formatted taxonomic counts
      counts <- get_melt(dataset, rank, "aurora")
      
      # Filter out samples removed from design matrix
      counts <- subset(counts, sample %in% meta$sample)
      
      # Remove Taxa with zero counts
      if (any(colSums(counts[,2:ncol(counts)]) == 0)){
        counts <- counts[,-(which(colSums(counts[,2:ncol(counts)]) == 0)+1)]
      }
      
      # Simplify taxa names
      saveRDS(data.frame(taxa = colnames(counts[,2:ncol(counts)]), colname = paste("Taxa",seq(1, ncol(counts)-1, by = 1),sep="_")), file = paste("data/taxa.names.",dataset,".",rank,".rds",sep=""))
      colnames(counts)[2:ncol(counts)] <- paste("Taxa",seq(1, ncol(counts)-1, by = 1),sep="_")
    }
    
    # Merge counts and metadata
    data <- inner_join(meta, counts, by=c("sample"="sample")) %>%
      drop_na()
    
    rownames(data) <- data$sample
    data$sample <- NULL
    data <- data[,c(2:ncol(counts),1)]
    data[,factor] <- factor(data[,factor], levels = c("1","2","3","4"))

    # Transform
    preProcValues <- preProcess(data, method = "range")
    dataTransformed <- predict(preProcValues, data)
    
    # source model
    model <- readRDS(paste("../models/regression/",file,sep=""))[4]
    
    # subset predictors in dataset to those present in model and vice versa
    if (mo == "RandomForest"){
      features <- gsub("trainingData.","",colnames(as.data.frame(model[[1]][13])))
    } else {
      features <- unlist(predictors(model))
      features <- gsub("\`","", features)
    }
    
    # remove those in the test set absent from model
    model.miss <- colnames(dataTransformed)[which(!(colnames(dataTransformed) %in% features))]
    dataTransformed <- cbind(dataTransformed[,which(colnames(dataTransformed) %in% features)],dataTransformed[,ncol(dataTransformed)])
    colnames(dataTransformed)[ncol(dataTransformed)] <- factor
    
    # add in dummy info for ASVs in the model, but absent from data
    foo <- data.frame(ASV = setdiff(features, taxa_names(p)), stringsAsFactors = F)
    row.names(foo) <- foo$ASV
    foo <- data.frame(t(foo), check.names = F, stringsAsFactors = F)
    foo[1:nrow(dataTransformed),] <- 0
    foo[] <- lapply(foo, function(x) as.numeric(as.character(x)))
    dataTransformed <- cbind(dataTransformed, foo)
    
    # run prediction
    predictions <- predict(model, dataTransformed)
    eval_metric <- confusionMatrix(as.factor(unlist(predictions)), dataTransformed[,factor])$overall[2]

    if (count == 1){
      results <- data.frame(mod = mo, see = seed, dataset = dataset, fact = factor, feat = feature_type, tax_rank = rank, kappa = eval_metric, datatype = permute_labels, stringsAsFactors = F)
      results2 <- rbind.fill(data.frame(mod = mo, see = seed, dataset = dataset, fact = factor, feat = feature_type, tax_rank = rank, test.missing = setdiff(features, taxa_names(p)), datatype = permute_labels, stringsAsFactors = F), data.frame(mod = mo, see = seed, dataset = dataset, fact = factor, feat = feature_type, tax_rank = rank, model.missing = model.miss, datatype = permute_labels, stringsAsFactors = F))
      count <- count + 1
    } else {
      results <- rbind(results, data.frame(mod = mo, see = seed, dataset = dataset, fact = factor, feat = feature_type, tax_rank = rank, kappa = eval_metric, datatype = permute_labels, stringsAsFactors = F))
      results2 <- rbind(results2, rbind.fill(data.frame(mod = mo, see = seed, dataset = dataset, fact = factor, feat = feature_type, tax_rank = rank, test.missing = setdiff(features, taxa_names(p)), datatype = permute_labels, stringsAsFactors = F), data.frame(mod = mo, see = seed, dataset = dataset, fact = factor, feat = feature_type, tax_rank = rank, model.missing = model.miss, datatype = permute_labels, stringsAsFactors = F)))        
    }
  }
}

colnames(results) <- c("model","seed","dataset","health.metric","feature_type","rank","kappa","test.set")
colnames(results2) <- c("model","seed","dataset","health.metric","feature_type","rank","missing.from.model","test.set","missing.from.testset")
results2 <- subset(results2, test.set == "real")

saveRDS(results, file = "data/aurora.classification.predictions.final.rds")
saveRDS(results2, file = "data/aurora.classification.missing.final.rds")
