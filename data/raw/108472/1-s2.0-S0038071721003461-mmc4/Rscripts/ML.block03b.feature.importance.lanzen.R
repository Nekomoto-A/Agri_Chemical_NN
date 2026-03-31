library(plyr)
library(phyloseq)
library(Hmisc)
library(tidyverse)
library(caret)
library(spaa)
library(parallel)
source('get_study.R')

fact <- "pH" # "yield.dry.wt" | "pH"

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

# define perm test function
perm_test <- function(file){
  
  ## Import necessary data (model, validation data (i.e. testData) count table)
  info <- unlist(parse_file(file))
  
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
  dataTransformed <- cbind(dataTransformed[,which(colnames(dataTransformed) %in% features)],dataTransformed[,ncol(dataTransformed)])
  colnames(dataTransformed)[ncol(dataTransformed)] <- fact
  
  # add in dummy info for ASVs in the model, but absent from data
  foo <- data.frame(ASV = setdiff(features, taxa_names(p)), stringsAsFactors = F)
  row.names(foo) <- foo$ASV
  foo <- data.frame(t(foo), check.names = F, stringsAsFactors = F)
  foo[1:nrow(dataTransformed),] <- 0
  foo[] <- lapply(foo, function(x) as.numeric(as.character(x)))
  dataTransformed <- cbind(dataTransformed, foo)
  
  ################################
  ## Perform Permutational Testing
  
  # -----------Use testMetric from held-out test data as the base-line for permutation testing--------->
  predictions <- predict(model, dataTransformed)[[1]]
  base_metric <- summary(lm(predictions ~ dataTransformed[,fact]))$r.squared
  otus <- colnames(dataTransformed[-which(colnames(dataTransformed) == fact)])
  
  # ----------- Get feature importance of non-correlated OTUs------------>
  
  # Output: the impact each non-correlated OTU makes in the prediction performance
  imp <- do.call('rbind', lapply(otus, function(i){
    full_permuted <- dataTransformed
    full_permuted[,i] <- sample(dataTransformed[,i])

    # Calculate the new metric
    predictions <- predict(model, full_permuted)[[1]]
    new_metric <- summary(lm(predictions ~ full_permuted[,fact]))$r.squared
    
    return(new_metric)
  }))

  ## Add back in ASV names and convert to a dataframe.
  imp <- as.data.frame(imp) %>%
    mutate(names=factor(otus)) 
  colnames(imp)[1] <- "Rsquared"
  
  # Order bulk of features
  imp <- as.data.frame(imp[order(imp[,1]),])
  colnames(imp) <- c("eval","names")

  # Save
  perm_results <- list(file, base_metric, imp)
  
  return(perm_results)
}

### 
### Permutational testing to evaluate contribution of each feature

## Capture all regression models
files <- list.files(path = "data/results/", pattern="*.rds")

## Subset to best models
bestmodels <- readRDS("../models/lanzen.correlations.compiled.rds")
bestmodels <- subset(bestmodels, lanzen.factor %in% c(fact))
bestmodels <- subset(bestmodels, health.metric %in% c("organic_matter_rating","pH_rating","active_carbon_rating","ace_soil_protein_index_rating","avg.rating"))
bestmodels <- subset(bestmodels, permuted == 0 & r > 0 & dataset == "minimal.norm" & rank == "ASV" & model == "L2LinearSVM")

## Subset list to best models
run_list <- vector(mode = "character")
for (file in files){
  find_me <- parse_file(file)
  find_me <- subset(bestmodels, model == find_me[[4]] & seed == find_me[[2]] & dataset == find_me[[3]] & health.metric == find_me[[5]] & rank == find_me[[6]])
  if (nrow(find_me) > 0){
    run_list <- append(run_list, file)
  }
}

## Import microbiome data
p <-  readRDS(file = "../microbiome.data/p_lanzen.final.rds")

# Read count data
if(any(taxa_sums(p) == 0)){
  p <- subset_taxa(p, taxa_sums(p) > 0)
}

# Prepare metadata and select only 
meta <- as.data.frame(as(sample_data(p), "matrix"), stringsAsFactors = F) %>%
  select(fact)
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
data[,fact] <- as.numeric(data[,fact])

# Repeat transformation
preProcValues <- preProcess(data, method = "range")
dataTransformed <- predict(preProcValues, data)

# run correlations
results <- mclapply(run_list, perm_test, mc.cores = 50)

# save backup
saveRDS(results, file = paste("data/lanzen.feature.importance.final.",fact,".rds",sep=""))


##################################
## Package results and save output

results <- readRDS(file = paste("../models/lanzen.feature.importance.final.",fact,".rds",sep=""))

# unpack parallelized data and ignore conditions where no model existed for job in joblist
for (n in 1:length(results)){
  
  # Parse results  
  x <- results[[n]]
  file <- x[[1]]
  baseline <- x[[2]]
  x <- as.data.frame(x[[3]])
  
  print(file)
  
  ## Import necessary data (model, validation data (i.e. testData) count table)
  info <- unlist(parse_file(file))
  x$experiment <- info[1]
  x$seed <- info[2]
  x$dataset <- info[3]
  x$model <- info[4]
  x$health.metric <- info[5]
  x$rank <- info[6]
  x$feature_type <- info[7]
  x$comboID <- paste(info[4], info[3], info[6], info[7],info[5], info[2], sep = ";")
  
  # Add information about base evaluation metric
  x$basemetric <- baseline
  x$ratio <- x$eval/x$basemetric
  x <- x[rev(order(x$ratio)),]
  x$file <- file
  
  if (n == 1){
    perm_results <- x 
  } else{
    perm_results <- rbind(x, perm_results)
  }    
}

# Set models that performed poorly (i.e. Kapp < 0) to zero
perm_results[which(perm_results$eval < 0), "ratio"] <- 0
perm_results[which(perm_results$eval < 0), "eval"] <- 0

# Save
saveRDS(perm_results, file = paste("data/lanzen.feature.importance.final.",fact,".rds",sep=""))