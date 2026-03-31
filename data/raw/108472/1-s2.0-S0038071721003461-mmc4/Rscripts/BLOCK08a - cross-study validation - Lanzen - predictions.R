library(phyloseq)
library(Hmisc)
library(tidyverse)
library(caret)
library(plyr)
library(parallel)
source('get_study.R')

###
### Prepare Lanzen data

firstpass <- "N"        # "Y" | "N" - based on whether to preprocess the Lanzen study data

if (firstpass == "Y"){
  ### Lanzen et al., 2015 (PRJEB9654): 10.3389/fmicb.2015.01321
  ### This was processed as part of the agro-ecoDB ;; Wilhelm et al., 2022
  lanzen <- readRDS(file = "../microbiome.data/p_lanzen.filtered.norm.rds")
  
  # Remove samples with zero counts
  lanzen <- prune_samples(names(sample_sums(lanzen)[which(!is.nan(sample_sums(lanzen)))]), lanzen)
  
  # add in full sample metadata (reduced for agro-ecoDB)
  lanz.meta <- read.csv(file = "../microbiome.data/lanzen.metadata.csv", header = T, stringsAsFactors = F)
  
  # remove samples without metadata
  lanzen <- prune_samples(lanz.meta$comboID, lanzen)
  lanz.meta <- subset(lanz.meta, comboID %in% sample_names(lanzen))
  
  # order and add to phyloseq
  lanz.meta <- lanz.meta[match(lanz.meta$comboID, sample_names(lanzen)),]
  row.names(lanz.meta) <- lanz.meta$comboID
  sample_data(lanzen) <- sample_data(lanz.meta[,c("penetrability","compaction","yield.dry.wt","yield.fresh.wt","potassium","pH","CO2.respiration","induced.respiration","organic_matter")])
  
  # save final dataset for consistent future use
  #saveRDS(lanzen, file = "../microbiome.data/p_lanzen.final.rds")
  
  # save melt for taxonomy based analysis
  x.lanzen <- psmelt(lanzen)
  saveRDS(x.lanzen, file = "../microbiome.data/melt.lanzen.final.rds")    
  
  # Tally all counts for each rank
  for (rank in c("Order","Family","Genus")){
    formula <- as.formula(paste("~",paste(c(rank,"Sample"), collapse = "+"), sep=""))
    foo <- ddply(x.lanzen, formula, summarise, Total.Abundance = sum(Abundance))
    foo <- reshape(foo, direction = "wide", idvar = "Sample", timevar = rank)
    colnames(foo) <- gsub("Total.Abundance.","",colnames(foo))
    colnames(foo)[1] <- "sample"
    
    saveRDS(foo, file = paste("data/melt.lanzen.final",rank,"rds", sep="."))
  }
} else {
  lanzen <- readRDS(file = "data/p_lanzen.final.rds")
}


###
### Make Predictions Using SH Modesl from Lanzen data

### Capture all regression models
files <- list.files(path = "../models/regression/", pattern="*.rds")

# remove classification models
files <- files[!grepl("category|tillage|soil_texture_class|root_pathogen_pressure|root_pathogen_pressure_rating|surface_hardness|surface_hardness_rating|subsurface_hardness|subsurface_hardness_rating", files)]

# not interested in anything but Genus and ASV
files <- files[!grepl("Order|Family", files)]

## Choose SH metrics to use (used all of them!)
factor_list <- c("avg.rating","surface_hardness_rating","subsurface_hardness_rating","organic_matter_rating","ace_soil_protein_index_rating","respiration_rating","active_carbon_rating","pH_rating","K_rating","DNA","water_capacity_rating","aggregate_stability_rating","P_rating","soil_texture_sand","soil_texture_silt","soil_texture_clay")
lanzen_factor <- "yield.dry.wt" # it does not matter which factor is used here. The 'predict' function requires observed data to compare with, but the prediction values are not based on this information. We will extract the prediction values and make the comparison separately.

## Run through each model and make predictions from Lanzen data
count <- 1
for (file in files){
  print(file)
  
  # parse model information
  seed <- unlist(strsplit(file, "_"))[4]
  dataset <- unlist(strsplit(file, "_"))[5]
  set.seed(as.numeric(seed))
  mo <- unlist(strsplit(file, "_"))[3]
  model_factor <- unlist(strsplit(file, paste(dataset,"",sep="_")))[2]
  model_factor <- gsub(".rds","",model_factor)
  
  # deal with special cases vis-a-vis taxonomy vs. ASV-based approaches
  if (grepl("Phylum|Class|Order|Family|Genus", file)){
    feature_type <- "Taxonomy"
    rank = unlist(strsplit(file, "_"))[length(unlist(strsplit(file, "_")))]
    rank <- gsub(".rds","",rank)
    model_factor <- gsub(paste("_",rank,sep=""),"",model_factor)
    
  } else {
    feature_type <- "ASV"
    rank <- "ASV"
  }
  
  # Make predictions fo each SH model
  if (model_factor %in% factor_list){
    
    # Import Lanzen data
    p <- lanzen
    
    # Prepare metadata and select only sample Id and diagnosis columns
    meta <- as.data.frame(as(sample_data(p), "matrix"), stringsAsFactors = F) %>%
      select(lanzen_factor) 
    meta$sample <- row.names(meta)
    
    if (feature_type == "ASV"){
      # Get rid of ASV with zero counts
      p <- subset_taxa(p, taxa_sums(p) > 0)
      
      # Read in OTU table and remove label and numOtus columns
      counts <- as.data.frame(as(t(otu_table(p)), "matrix"))
      
      # add in sample info
      counts$sample <- row.names(counts)
      
    } else {
      
      # Read in pre-formatted taxonomic counts
      counts <- readRDS(file = paste("../microbiome.data/melt.lanzen.final",rank,"rds", sep="."))
      
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
    data[,lanzen_factor] <- as.numeric(data[,lanzen_factor])
    
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
    
    # transform data the same as during preparation for RF modeling
    model.miss <- colnames(dataTransformed)[which(!(colnames(dataTransformed) %in% features))]
    dataTransformed <- cbind(dataTransformed[,which(colnames(dataTransformed) %in% features)],dataTransformed[,ncol(dataTransformed)])
    colnames(dataTransformed)[ncol(dataTransformed)] <- lanzen_factor
    
    # add in dummy info for ASVs in the model, but absent from data
    foo <- data.frame(ASV = setdiff(features, taxa_names(p)), stringsAsFactors = F)
    row.names(foo) <- foo$ASV
    foo <- data.frame(t(foo), check.names = F, stringsAsFactors = F)
    foo[1:nrow(dataTransformed),] <- 0
    foo[] <- lapply(foo, function(x) as.numeric(as.character(x)))
    dataTransformed <- cbind(dataTransformed, foo)
    
    # run prediction
    predictions <- predict(model, dataTransformed)

    # save results
    if (count == 1){
      results <- data.frame(mod = mo, see = seed, dataset = dataset, feat = feature_type, tax_rank = rank, lanzen.sampleID = row.names(meta), prediction.data = unlist(predictions), model.factor = model_factor, stringsAsFactors = F)
      results2 <- rbind.fill(data.frame(mod = mo, see = seed, dataset = dataset, fact = model_factor, feat = feature_type, tax_rank = rank, test.missing = setdiff(features, taxa_names(p)), stringsAsFactors = F), data.frame(mod = mo, see = seed, dataset = dataset, fact = model_factor, feat = feature_type, tax_rank = rank, model.missing = model.miss, stringsAsFactors = F))
      count <- count + 1
    } else {
      results <- rbind(results, data.frame(mod = mo, see = seed, dataset = dataset, feat = feature_type, tax_rank = rank, lanzen.sampleID = row.names(meta), prediction.data = unlist(predictions), model.factor = model_factor, stringsAsFactors = F))
      results2 <- rbind(results2, rbind.fill(data.frame(mod = mo, see = seed, dataset = dataset, fact = model_factor, feat = feature_type, tax_rank = rank, test.missing = setdiff(features, taxa_names(p)), stringsAsFactors = F), data.frame(mod = mo, see = seed, dataset = dataset, fact = model_factor, feat = feature_type, tax_rank = rank, model.missing = model.miss, stringsAsFactors = F)))
      
    }
  }
}

# Save Output
colnames(results) <- c("model","seed","dataset", "feature_type","rank","lanzen.sampleID","prediction","model_factor")
colnames(results2) <- c("model","seed","dataset","health.metric","feature_type","rank","missing.from.model","missing.from.testset")

saveRDS(results, file = "data/lanzen.predictions.rds")
saveRDS(results2, file = "data/lanzen.missing.features.final.rds")


################
################ Correlate predictions with all Lanzen data

## Define function to perform correlations
corr_me <- function(job){
  mod <- unlist(strsplit(job, ";"))[1]
  s <- unlist(strsplit(job, ";"))[2]
  data <- unlist(strsplit(job, ";"))[3]
  r <- unlist(strsplit(job, ";"))[4]
  lanzen <- unlist(strsplit(job, ";"))[5]
  metric <- unlist(strsplit(job, ";"))[6]
  
  # subset to lanzen factor
  x <- subset(y, model == mod & seed == s & dataset == data & rank == r & model_factor == metric)
  
  # merge with normalized lanzen observed data
  foo <- meta[,c(lanzen, "lanzen.sampleID")]
  colnames(foo) <- c("observed","lanzen.sampleID")
  x <- merge(x, foo, by = "lanzen.sampleID")
  
  if (nrow(x) > 0){
    for (permute_labels in c(0,1)){
      if (permute_labels == 1){
        set.seed(10)
        x$observed <- sample(x$observed)
      }
      
      Rsqr <- summary(lm(observed ~ prediction, data = x))$r.squared
      corr.data <- rcorr(x$observed, x$prediction,type= "pearson")
      mse <- mean((summary(lm(observed ~ prediction, data = x))$residuals)^2)
      
      if (permute_labels == 0){
        results <- data.frame(health.metric = metric, lanzen.factor = lanzen, model = mod, seed = s, dataset = data, rank = r, R2 = Rsqr, MSE = mse, r = corr.data$r[1,2], p = corr.data$P[1,2], permuted = permute_labels, stringsAsFactors = F)
      } else {
        results <- rbind(results, data.frame(health.metric = metric, lanzen.factor = lanzen, model = mod, seed = s, dataset = data, rank = r, R2 = Rsqr, MSE = mse, r = corr.data$r[1,2], p = corr.data$P[1,2], permuted = permute_labels, stringsAsFactors = F))
      }
    }
  } else {
    results <- "no model"
  }
  
  return(results)
}

## Normalize all Lanzen observed data (to match prediction ranges)
lanzen_factors <- c("yield.dry.wt","yield.fresh.wt","penetrability","compaction","organic_matter","CO2.respiration","induced.respiration","pH","potassium")

# Import data
p <- readRDS(file = "data/p_lanzen.final.rds")

# Prepare Lanzen data 
meta <- as.data.frame(as(sample_data(p), "matrix"), stringsAsFactors = F) 
meta <- meta[,lanzen_factors]

# Normalize
preProcValues <- preProcess(meta, method = "range")
meta <- predict(preProcValues, meta)
meta$lanzen.sampleID <- row.names(meta)

# Import predictions
y <- readRDS(file = "../models/lanzen.predictions.rds")

## Run through all pairings and perform correlation (use mcapply to use multiple cores)
## Run through each factor in lanzen study and perform correlation for all seeds * both models * ranks/features * actual/permutation
models <- unique(y$model)
seeds <- unique(y$seed)
datasets <- unique(y$dataset)
ranks <- unique(y$rank)
metrics <- unique(y$model_factor)

for (lanzen in lanzen_factors){

  ## Run correlations for all seeds * models * ranks * dataset * SH metric ... in parallel
  # make list of jobs for mcapply script
  joblist <- vector(mode="character")
  
  for (mod in models){
    for (s in seeds){
      for (data in datasets){
        for (r in ranks){
          for (metric in metrics){
            joblist <- append(joblist, paste(mod,s,data,r,lanzen,metric,sep=";"))
          }
        }
      }
    }
  }

  # remove classification models
  joblist <- joblist[grepl("minimal.norm", joblist)]
  
  # run correlations
  results <- mclapply(joblist, corr_me, mc.cores = 50)

  # unpack parallelized data and ignore conditions where no model existed for job in joblist
  count <- 1
  for (n in 1:length(results)){
    x <- results[[n]]

    if (x != "no model"){
      if (count == 1){
        output <- x 
        count <- count + 1
      } else{
        output <- rbind(x, output)
      }    
    }    
  }

  # save final
  saveRDS(output, file = paste("../models/SH.lanzen.",lanzen,".correlations.rds",sep=""))
  
}

## compile results
count <- 1
for (lanzen in lanzen_factors){
  output <- readRDS(file = paste("../models/SH.lanzen.",lanzen,".correlations.rds",sep=""))
  
  if (count == 1){
    final <- output
    count <- count + 1
  } else {
    final <- rbind(final, output)
  }
}

saveRDS(final, file = "../models/lanzen.correlations.compiled.rds")