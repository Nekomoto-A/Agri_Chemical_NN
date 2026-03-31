library(plyr)
library(phyloseq)
library(Hmisc)
library(tidyverse)
library(caret)
library(spaa)
library(parallel)
source('get_study.R')
source('get_melt.R')
source('down_sample_by_geogroup.R')

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
### Capture all regression models
files <- list.files(path = "../models/classification/", pattern="*.rds")
files <- append(files, list.files(path = "../models/regression/", pattern="*.rds"))

###
### Subset to best models
bestmodels <- readRDS(file = "../models/bestmodels.SH.rds") # from BLOCK04b -- Targeting best models in SH data

run_list <- vector(mode = "character")
for (file in files){
  find_me <- parse_file(file)
  find_me <- subset(bestmodels, model == find_me[[4]] & seed == find_me[[2]] & dataset == find_me[[3]] & factor == find_me[[5]] & rank == find_me[[6]])
  if (nrow(find_me) > 0){
    run_list <- append(run_list, file)
  }
}

# Parallel processing occassionally resutled in some jobs throwing an error (so the script was re-run)
#completed <- readRDS(file = "data/completed.feature.selection.rds")
#run_list <- run_list[-which(run_list %in% completed)]

### 
### Permutational testing to evaluate contribution of each feature

# define perm test function
perm_test <- function(file){
  
  ## Import necessary data (model, validation data (i.e. testData) count table)
  info <- unlist(parse_file(file))
  
  run.name <- info[1]
  seed <- info[2]
  set.seed(as.numeric(seed))
  dataset <- info[3]
  mo <- info[4]
  factor <- info[5]
  rank <- info[6]
  feature_type <- info[7]

  # assign evaluation method
  if (grepl("category", factor) | grepl("soil_texture_class", factor) | grepl("tillage", factor)){
    met <- "Accuracy"
  } else {
    met <- "Rsquared"
  }
  
  ## Import and prepare microbiome data (consistent with treatment during training)
  p <- get_study(dataset, "full")   
  
  # Call 'Strip till' as Tillage level 1
  sample_data(p)[which(sample_data(p)[,"tillage"] == "Strip Till"),"tillage"] <- "1"
  
  # Prepare metadata and select only sample Id and diagnosis columns
  meta <- as.data.frame(as(sample_data(p), "matrix"), stringsAsFactors = F) %>%
    select(factor)
  meta$sample <- row.names(meta)
  
  # Read count data
  if (feature_type == "ASV"){
    # Downsample the number of samples from any management geogroup to max 10
    p <- prune_samples(readRDS(file = "../microbiome.data/ML.sample.set.rds"), p)
    
    # Discard Features (ASV | Taxa) that occur in fewer than 10 samples
    present_absent <- otu_table(p) 
    present_absent[present_absent > 0] <- 1
    p <- prune_taxa(rownames(present_absent)[which(rowSums(present_absent) >= 10)], p)
    
    # Get rid of ASV with zero counts
    if(any(taxa_sums(p) == 0)){
      p <- subset_taxa(p, taxa_sums(p) > 0)
    }
    
    # make count table
    counts <- as.data.frame(as(t(otu_table(p)), "matrix"))
    
    # add in sample info
    counts$sample <- row.names(counts)
    
  } else {
    
    # Read in pre-formatted taxonomic counts
    counts <- get_melt(dataset, rank, "full")
    
    # Filter out samples removed from design matrix
    counts <- subset(counts, sample %in% meta$sample)
    
    # Filter out samples not used in ML
    counts <- subset(counts, row.names(counts) %in% readRDS(file = "../microbiome.data/ML.sample.set.rds"))
    
    # Discard Features (ASV | Taxa) that occur in fewer than 10 samples
    present_absent<-counts[,2:ncol(counts)]
    present_absent[present_absent > 0] <- 1
    counts <- counts[,-which(colnames(counts) %in% names(which(colSums(present_absent) < 10)))]
    
    # Remove Taxa with zero counts
    if (any(colSums(counts[,2:ncol(counts)]) == 0)){
      counts <- counts[,-(which(colSums(counts[,2:ncol(counts)]) == 0)+1)]
    }
    
    # Simplify taxa names
    saveRDS(data.frame(taxa = colnames(counts[,2:ncol(counts)]), colname = paste("Taxa",seq(1, ncol(counts)-1, by = 1),sep="_")), file = paste("../microbiome.data/taxa.names.",dataset,".",rank,".rds",sep=""))
    colnames(counts)[2:ncol(counts)] <- paste("Taxa",seq(1, ncol(counts)-1, by = 1),sep="_")
  }

  # Merge counts and metadata
  data <- inner_join(meta, counts, by=c("sample"="sample")) %>%
    select(-sample) %>%
    drop_na()
  data <- data[,c(2:ncol(counts),1)]
  
  # Repeat transformation
  preProcValues <- preProcess(data, method = "range")
  dataTransformed <- predict(preProcValues, data)
 
  # Import model
  if (grepl("category",file)){
    model <- readRDS(paste("../model/classification/",file,sep=""))[4]
  } else {
    model <- readRDS(paste("../model/regression/",file,sep=""))[4]
  }

  # Subset to testset
  trainingset <- row.names(as.data.frame(model[[1]][13]))
  testset <- setdiff(row.names(dataTransformed), trainingset)
  testTransformed  <- dataTransformed[which(row.names(dataTransformed) %in% testset),]
  
  if (grepl("category", file)){
    testTransformed[,factor] <- as.factor(testTransformed[,factor])
  } else {
    testTransformed[,factor] <- as.numeric(testTransformed[,factor])
  }
  
  # Consolidate model and testing feature sets (subset predictors in dataset to those present in model and vice versa)
  if (mo == "RandomForest"){
    features <- gsub("trainingData.","",colnames(as.data.frame(model[[1]][13])))
  } else {
    features <- unlist(predictors(model))
    features <- gsub("\`","", features)
  }
  
  # remove those in the test set absent from model
  testTransformed <- cbind(testTransformed[,which(colnames(testTransformed) %in% features)],testTransformed[,ncol(testTransformed)])
  colnames(testTransformed)[ncol(testTransformed)] <- factor
  
  # add in dummy info for ASVs in the model, but absent from data
  if (feature_type == "ASV"){
    foo <- data.frame(ASV = setdiff(features, taxa_names(p)), stringsAsFactors = F)
    row.names(foo) <- foo$ASV
    foo <- data.frame(t(foo), check.names = F, stringsAsFactors = F)
    foo[1:nrow(testTransformed),] <- 0
    foo[] <- lapply(foo, function(x) as.numeric(as.character(x)))
    testTransformed <- cbind(testTransformed, foo)
  } else {
    foo <- data.frame(Taxa = setdiff(features, colnames(counts)), stringsAsFactors = F)
    row.names(foo) <- foo$Taxa
    foo <- data.frame(t(foo), check.names = F, stringsAsFactors = F)
    foo[1:nrow(testTransformed),] <- 0
    foo[] <- lapply(foo, function(x) as.numeric(as.character(x)))
    testTransformed <- cbind(testTransformed, foo)
  }
  
  ################################
  ## Perform Permutational Testing

  # -----------Use testMetric from held-out test data as the base-line for permutation testing--------->
  
  ## ROC was not used b/c all factors contained > 2 variables
  if (met == "Accuracy"){
    testTransformed[,factor] <- gsub("\\(20,40\\]","cat1",testTransformed[,factor])
    testTransformed[,factor] <- gsub("\\(40,60\\]","cat2",testTransformed[,factor])
    testTransformed[,factor] <- gsub("\\(60,80\\]","cat3",testTransformed[,factor])
    testTransformed[,factor] <- gsub("\\(80,100\\]","cat4",testTransformed[,factor])
    testTransformed[,factor] <- as.factor(testTransformed[,factor])
    
    predictions <- predict(model, testTransformed)[[1]]
    base_metric <- confusionMatrix(predictions, testTransformed[,factor])$overall[2]
  } else if (met == "Rsquared"){
    predictions <- predict(model, testTransformed)[[1]]
    base_metric <- summary(lm(predictions ~ testTransformed[,factor]))$r.squared
  }

  # ----------- Remove any correlated OTUs-------------------->
  
  ## Get Correlations (ASV with spearman |r| > 0.9)
  if (feature_type == "ASV"){
    
    # try to import precalculated correlation table
    corr <- try(readRDS(paste("../microbiome.data/corr_",dataset,".rds",sep="")), silent = TRUE)
        
    # If the file does not exist, calculate it for the first time
    if (class(corr) == "try-error") {
      
      corr <- try(readRDS(paste("../microbiome.data/raw.corr_",dataset,".rds",sep="")), silent = TRUE)
      
      if (class(corr) == "try-error") {
        row.names(counts) <- counts$sample
        counts <- counts[,-1]
        corr <- rcorr(as.matrix(counts), type="spearman")
        
        # save output for ASV counts given the substantial computational time
        saveRDS(corr, paste("data/raw.corr_",dataset,".rds",sep=""))
      } else {
        # subset to correlated ASVs
        r <- dist2list(as.dist(corr$r))
        p <- dist2list(as.dist(corr$P))
        r <- subset(r, abs(value) > 0.9)
        p <- subset(p, value < 0.01)
        
        # get list of all correlated features
        if (nrow(r) > 0){
          count <- 1
          for (i in 1:nrow(r)){
            if (any(p$col == r$col[i] & p$row == r$row[i])){
              if (count == 1){
                corr <- data.frame(feature1 = as.character(r$col[i]), feature2 = as.character(r$row[i]), stringsAsFactors = F)  
                count <- count + 1
              } else {
                corr <- rbind(corr, data.frame(feature1 = as.character(r$col[i]),feature2 = as.character(r$row[i]), stringsAsFactors = F))
              }
            }
          }
          
          # Remove pairs
          corr <- corr[!duplicated(apply(corr[1:2], 1, function(x) toString(sort(x)))),]
          
        } else {
          corr <- "none" 
        }
        
        # Save output
        saveRDS(corr, paste("data/corr_",dataset,".rds",sep=""))
      }
    }
  } else {
    row.names(counts) <- counts$sample
    counts <- counts[,-1]
    corr <- rcorr(as.matrix(counts), type="spearman")
    r <- dist2list(as.dist(corr$r))
    p <- dist2list(as.dist(corr$P))
    r <- subset(r, value > 0.9 | value < -0.9)
    p <- subset(p, value < 0.01)
    
    # get list of all correlated features
    if (nrow(r) > 0){
      count <- 1
      for (i in 1:nrow(r)){
        if (any(p$col == r$col[i] & p$row == r$row[i])){
          if (count == 1){
            corr <- data.frame(feature1 = as.character(r$col[i]), feature2 = as.character(r$row[i]), stringsAsFactors = F)  
            count <- count + 1
          } else {
            corr <- rbind(corr, data.frame(feature1 = as.character(r$col[i]),feature2 = as.character(r$row[i]), stringsAsFactors = F))
          }
        }
      }
      
      # Remove pairs
      corr <- corr[!duplicated(apply(corr[1:2], 1, function(x) toString(sort(x)))),]
    } else {
      corr <- "none"
    }
  }

  ## Remove correlated ASVs
  if (any(corr != "none")){
    remove_me <- unique(c(corr$feature1, corr$feature2))
    remove_me <- remove_me[which(remove_me %in% colnames(testTransformed))]
    
    non_correlated_otus <- testTransformed %>%
      select(-remove_me) %>%
      select(-factor) %>%
      colnames()
  } else {
    non_correlated_otus <- testTransformed %>% select(-factor) %>% colnames()
  }
  
  # ----------- Get feature importance of non-correlated OTUs------------>
  
  # Output: the impact each non-correlated OTU makes in the prediction performance
  non_corr_imp <- do.call('rbind', lapply(non_correlated_otus, function(i){
    
    full_permuted <- testTransformed
    full_permuted[,i] <- sample(testTransformed[,i])
    
    # Predict the outcome with the one-feature-permuted test dataset
    # Calculate the new metric

    if (met == "Accuracy"){
      predictions <- predict(model, full_permuted)[[1]]
      new_metric <- confusionMatrix(predictions, full_permuted[,factor])$overall[2]
    } else if (met == "Rsquared"){
      predictions <- predict(model, full_permuted)[[1]]
      new_metric <- summary(lm(predictions ~ full_permuted[,factor]))$r.squared
    }
    
    return(new_metric)
  }))

  ## Add back in ASV names and convert to a dataframe.
  if (met == "Accuracy"){
    non_corr_imp <- as.data.frame(non_corr_imp) %>%
      mutate(names=factor(non_correlated_otus)) 
    colnames(non_corr_imp)[1] <- "Kappa"
  } else if (met == "Rsquared"){
    non_corr_imp <- as.data.frame(non_corr_imp) %>%
      mutate(names=factor(non_correlated_otus)) 
    colnames(non_corr_imp)[1] <- "Rsquared"
  }
  
  # ----------- Get feature importance of correlated OTUs -------------->
  if (any(corr != "none")){
    
    # Have each OTU in a group with all the other OTUs its correlated with
    # Each OTU should only be in a group once.
    non_matched_corr <- corr %>% filter(!feature1 %in% feature2) %>% group_by(feature1)
    
    if (nrow(non_matched_corr) > 0){
      
      # We use that tidyverse group_split to create a list of the OTUs that are grouped
      split <- group_split(non_matched_corr)
      
      # We want groups of OTUs all together and no repetition
      groups <- lapply(1:length(split), function(i){
        grouped_corr_otus <- split[[i]][2] %>%
          add_case(feature2=unlist(unique(split[[i]][1])))
        return(grouped_corr_otus)
      })
      
      # The list still had dataframes is them. We want the list entries to be lists as well
      groups_list <- map(groups[1:length(split)], "feature2")
      groups_list_sorted <- map(groups_list[1:length(split)], sort)
    } else {
      groups_list_sorted <- list(c(corr[1,1],corr[1,2]))
    }
    
    # Permute the grouped OTUs together and calculate AUC change
    
    corr_imp <- do.call('rbind', lapply(groups_list_sorted, function(i){
      full_permuted_corr <- testTransformed
      full_permuted_corr[,unlist(groups_list_sorted[i])] <- sample(testTransformed[,unlist(groups_list_sorted[i])])
      
      # Calculate the new metric
      if (met == "Accuracy") {
        predictions <- predict(model, full_permuted_corr)[[1]]
        new_metric <- confusionMatrix(predictions, full_permuted_corr[,factor])$overall[2]
        
      } else if (met == "Rsquared") {
        predictions <- predict(model, full_permuted_corr)[[1]]
        new_metric <- summary(lm(predictions ~ full_permuted_corr[,factor]))$r.squared
      }

      list <- list(new_metric, unlist(i))
      return(list)
      
    }))
    
    # Save non correlated results in a dataframe
    corr_imp_appended <- as.data.frame(corr_imp) 
    
    # Unlist percent metric change to save it as a csv later
    results <- corr_imp_appended %>%
      mutate(new_metric=unlist(corr_imp_appended$V1))
    
    # Only keep the columns that are not all NA
    corr_imp <- results %>%
      select(-V1) 
  } else {
    corr_imp <- "none"
  }
  
  # Order bulk of features
  non_corr_imp <- as.data.frame(non_corr_imp[rev(order(non_corr_imp[,1])),])
  colnames(non_corr_imp) <- c("eval","names")
  
  # Add back in taxa names
  if (feature_type != "ASV"){
    names <- readRDS(file = paste("../microbiome.data/taxa.names.",dataset,".",rank,".rds",sep=""))  
    non_corr_imp <- merge(non_corr_imp, names, by.x="names",by.y="colname")
    non_corr_imp$names <- NULL
    colnames(non_corr_imp) <- c("eval","names")
    non_corr_imp <- non_corr_imp[rev(order(non_corr_imp$eval)),]
  } 
  
  # Order correlated features
  if (corr_imp != "none"){
    for (i in 1:nrow(corr_imp)){
      corr_imp[i,1] <- as.character(paste(unlist(corr_imp[i,1]),collapse = ";"))
    }
    corr_imp$V2 <- as.character(corr_imp$V2)
    corr_imp <- as.data.frame(corr_imp[rev(order(corr_imp[,1])),])
    corr_imp <- unique(corr_imp)
    
    colnames(corr_imp) <- c("names","eval")
    
    ## going to have to get names manually for this subset.
  }
  
  # Save
  perm_results <- list(file, base_metric, non_corr_imp, corr_imp)
  
  return(perm_results)
}

# run correlations
results <- mclapply(run_list, perm_test, mc.cores = 50)

# save backup
saveRDS(results, file = "data/feature.selection.final.rds")


##################################
## Package results and save output

# Import results from above
results <- readRDS(file = "../models/feature.selection.final.rds")

# unpack parallelized data and ignore conditions where no model existed for job in joblist
count <- 1
for (n in 1:length(results)){
  
  # Extract individual permutational testing run
  x <- results[[n]]

  # Parse results  
  if (length(x) != 1){
    file <- x[[1]]
    baseline <- x[[2]]
    non_corr <- as.data.frame(x[[3]])
    non_corr$group <- "non-correlated"
    
    print(file)
    if (x[[4]] != "none"){
      corr <- as.data.frame(x[[4]])
      corr$group <- "correlated"
      x <- rbind(non_corr, corr)
    } else {
      x <- non_corr
    }
    
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
    x <- x[order(x$ratio),]
    x$file <- file
     
    if (count == 1){
      perm_results <- x 
      count <- count + 1
    } else{
      perm_results <- rbind(x, perm_results)
    }    
  }
}


# Remove classification models that performed poorly (i.e. Kappa < 0)
perm_results <- subset(perm_results, eval > 0 & basemetric > 0)

# Save
saveRDS(perm_results, file = "../models/compiled.feature.selection.results.final.rds")
