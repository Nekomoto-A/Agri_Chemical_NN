# Main script from Topcuoglu et al., 2020

deps = c("dplyr", "tictoc", "caret" ,"rpart", "randomForest", "kernlab","LiblineaR", "pROC", "tidyverse","Hmisc","spaa","Hmisc")

for (dep in deps){
  #install.packages(as.character(dep), dependencies=TRUE)
  library(dep, verbose=FALSE, character.only=TRUE)
}

# Import custom code from Topcuoglu et al., 2020 
source('ML.function01.model.evaluation.get_results.R')   # get_results()
source('ML.function02.regularization.tuning_grid.R')     # tuning_grid()
source('ML.block01.pipeline.R')                          # has MAIN pipeline function defined here 
source('get_study.R')
source('get_melt.R')
source('down_sample_by_geogroup.R')

#############
## User Input

manual <- "T" # "T" | "F"
first_pass <- "F"  # Set to true for only the first use of script in a new project directory. Setting to False saves time where possible

# Command-line input (to feed the seed from a Python loop in the command line)
input <- commandArgs(trailingOnly=TRUE)
seed <- as.numeric(input[1])
run.name <- as.character(input[2])
factor <- as.character(input[3])
dataset <- as.character(input[4])
feature_type <- as.character(input[5])
rank <- as.character(input[6])
model <- as.character(input[7])

# Manually set parameters for trouble-shooting
if (manual == "T"){
  seed <- "0"
  dataset <- "minimal.norm"    # "minimal", "minimal.rare", "minimal.norm", "filtered", "filter.rare", "filter.css" or "filter.norm"
  feature_type <- "ASV"  # "ASV" | "Taxonomy" | "Topcuoglu"
  rank <- "Order"             # If feature_type == "Taxonomy, the following options can be specified: "Phylum","Class","Order","Family","Genus"
  factor <- "health.category"      # "health.category", "avg.rating"
  model <- "L2LinearSVM"    # "L2LinearSVM" (regression | classification) | "RandomForest" (regression | classification) | "L2LogisticRegression" (classification only)
}

# Automate choice of classification/regression (this will break when used in multithreading. Hopefully the user will catch this beforehand and hardcode their ML_approach.)
if (grepl("category",factor)){
  ML_approach <- "classification" 
} else if (grepl("rating",factor)){
  ML_approach <- "regression"
} else if (factor %in% c("DNA","soil_texture_sand","soil_texture_silt","soil_texture_clay")){
  ML_approach <- "regression"
} else if (factor %in% c("tillage","soil_texture_class")){
  ML_approach <- "classification"
} else {
  print("You will need to specify whether you wish to perform regression- or classification-based supervised machine learning.")
  print(factor)
  
  writeLines("How to proceed?")
  switch(menu(c("classification","regression","exit")), ML_approach <- "classification", ML_approach <- "regression", ML_approach <- "exit")
}

if (ML_approach == "exit"){
  print("You have chosen to exit the runML script.")
  quit()
}

###########################
## Part I: Data Preparation

## Import data
if (feature_type != "Topcuoglu"){
  
  # Import soil health microbiome data
  p <- get_study(dataset, "full")   
  
  # Call 'Strip till' as Tillage level 1
  sample_data(p)[which(sample_data(p)[,"tillage"] == "Strip Till"),"tillage"] <- "1"
  
  # Downsample the number of samples from any management geogroup to max 10
  if (first_pass == "Y"){
    cap <- table(sample_data(p)[,"manage_group"])[which(table(sample_data(p)[,"manage_group"]) > 10)]  
    p_save <- subset_samples(p, !(manage_group %in% names(cap)))
    
    if (length(cap) > 0){
      for (site in names(cap)){
        p_subset <- subset_samples(p, manage_group == site)
        p_subset <- prune_samples(sample(sample_names(p_subset), 10, replace=F), p_subset)
        p_save <- merge_phyloseq(p_save, p_subset)
      }
    }
    
    p <- p_save
    rm(p_save)
    saveRDS(sample_names(p), file = "../microbiome.data/ML.sample.set.rds") # (for use in BLOCK02 and BLOCK01b)
    
  } else {
    p <- prune_samples(readRDS(file = "../microbiome.data/ML.sample.set.rds"), p)
  }

  # Prepare metadata and select only sample Id and diagnosis columns
  meta <- as.data.frame(as(sample_data(p), "matrix"), stringsAsFactors = F) %>%
    select(factor)
  meta$sample <- row.names(meta)
  
  # Extract Count Data
  if (feature_type == "ASV"){
    # Discard Features (ASV | Taxa) that occur in fewer than 10 samples
    present_absent<-otu_table(p) 
    present_absent[present_absent > 0] <- 1
    p <- prune_taxa(rownames(present_absent)[which(rowSums(present_absent) >= 10)], p)
    
    # Get rid of ASV with zero counts
    p <- subset_taxa(p, taxa_sums(p) > 0)
    
    # Read in OTU table and remove label and numOtus columns
    counts <- as.data.frame(as(t(otu_table(p)), "matrix"))
    
    # add in sample info
    counts$sample <- row.names(counts)
    
  } else {
    
    # Read in pre-formatted taxonomic counts
    counts <- get_melt(dataset, rank, "full")
    
    # Filter out samples removed from design matrix
    counts <- subset(counts, sample %in% meta$sample)
    
    # Discard Features (ASV | Taxa) that occur in fewer than 10 samples
    present_absent<-counts[,2:ncol(counts)]
    present_absent[present_absent > 0] <- 1
    counts <- counts[,-which(colnames(counts) %in% names(which(colSums(present_absent) < 10)))]
    
    # Remove Taxa with zero counts
    if (any(colSums(counts[,2:ncol(counts)]) == 0)){
      counts <- counts[,-(which(colSums(counts[,2:ncol(counts)]) == 0)+1)]
    }
    
    # Simplify taxa names
    saveRDS(data.frame(taxa = colnames(counts[,2:ncol(counts)]), colname = paste("Taxa",seq(1, ncol(counts)-1, by = 1),sep="_")), file = paste("data/taxa.names.",dataset,".",rank,".rds",sep=""))
    colnames(counts)[2:ncol(counts)] <- paste("Taxa",seq(1, ncol(counts)-1, by = 1),sep="_")
  }

  #saveRDS(p, file = "data/minimal.ASV.datset.rds") for use in BLOCK07c and BLOCK08c
  
  # Merge counts and metadata
  data <- inner_join(meta, counts, by=c("sample"="sample")) %>%
    drop_na()
  rownames(data) <- data$sample
  data$sample <- NULL
  data <- data[,c(2:ncol(counts),1)]

  # Fix naming to conform for training function in caret
  if (grepl("category", factor)){
    data[which(data[,factor] == "(20,40]"),factor] <- "cat1"
    data[which(data[,factor] == "(40,60]"),factor] <- "cat2"
    data[which(data[,factor] == "(60,80]"),factor] <- "cat3"
    data[which(data[,factor] == "(80,100]"),factor] <- "cat4"
    
  }
  
  # Perform correlation for use in 'function03'
  if (feature_type == "ASV"){
    corr <- try(readRDS(paste("../microbiome.data/raw.corr_",dataset,".rds",sep="")), silent = TRUE)
    
    if (class(corr) == "try-error") {
      
      # If the file does not exist, calculate it for the first time
      row.names(counts) <- counts$sample
      counts <- counts[,-1]
      corr <- rcorr(as.matrix(counts), type="spearman")
      r <- dist2list(as.dist(corr$r))
      p <- dist2list(as.dist(corr$P))
      r <- subset(r, value > 0.9 | value < -0.9)
      p <- subset(p, value < 0.01)
      
      # save output for ASV counts given the substantial computational time
      saveRDS(corr, paste("data/raw.corr_",dataset,".rds",sep=""))
      
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
} else {
  meta <- read.delim('../microbiome.data/topcuoglu_example.metadata.tsv', header=T, sep='\t') %>%
    select(sample, Dx_Bin, fit_result)
  
  shared <- read.delim('../microbiome.data/topcuoglu_example.counts.tsv', header=T, sep='\t') %>%
    select(-label, -numOtus)
  
  data <- inner_join(meta, shared, by=c("sample"="Group")) %>%
    mutate(dx = case_when(
      Dx_Bin== "Adenoma" ~ "normal",
      Dx_Bin== "Normal" ~ "normal",
      Dx_Bin== "High Risk Normal" ~ "normal",
      Dx_Bin== "adv Adenoma" ~ "cancer",
      Dx_Bin== "Cancer" ~ "cancer"
    )) %>%
    select(-sample, -Dx_Bin, -fit_result) %>%
    drop_na()
  
  factor <- "dx"
  
  # ----------- Read in the correlation matrix of full dataset---------->
  
  # From manuscript (Topcuoglu et al., 2020):
  # We grouped OTUs that had a perfect correlation with each other; however, we could reduce the correlation
  # threshold to further investigate the relationships among correlated features. By our approach,
  # we identified 432 OTUs out of 6,920 that had perfect Spearman correlations with at least one other OTU (corr = 1; p <0.01).
  # The decision to establish correlation thresholds is left to researchers to implement for their own
  # analyses. Regardless of the threshold, undestanding the correlation structures within the data is
  # critical to avoid misinterpreting the models.
  
  corr <- read.csv("../microbiome.data/sig_flat_corr_matrix.csv", stringsAsFactors = F, header = T) %>% select(-p, -cor)
  colnames(corr) <- c("feature1","feature2")
}

# Make response variable a factor
if (ML_approach == "classification"){
  data[,factor] <- factor(data[,factor])
} else {
  data[,factor] <- as.numeric(data[,factor])
}

data <- data[sample(1:nrow(data)), ]

############################
## Part II: Run ML Workflow

set.seed(seed)

# Start walltime for running model
tic("model")

# Run the model
get_results(data, model, seed, factor, run.name)  # function01

# Stop walltime for running model
secs <- toc()

# Save elapsed time
walltime <- secs$toc-secs$tic

# Save wall-time
if (feature_type == "Taxonomy"){
  save_name <- paste("../models/walltime/walltime_", run.name, "_", model, "_", seed, "_", dataset, "_", factor, "_", rank, ".csv", sep="")
} else {
  save_name <- paste("../models/walltime/walltime_", run.name, "_", model, "_", seed, "_", dataset, "_", factor, ".csv", sep="")
}
write.csv(walltime, file=save_name, row.names=F)