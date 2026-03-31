library(phyloseq)
library(plyr)
library(ggplot2)
library(reshape2)
source("get_study.R")

# Run Indicator analysis on all SH categories
indic <- function(factor){
  indval <- multipatt(get(paste(factor,"_counts",sep="")), get(paste(factor,"_classes",sep="")), control = how(nperm=999))
}

# User Input
physical <- c("water_capacity_rating", "surface_hardness_rating", "subsurface_hardness_rating", "aggregate_stability_rating")  # Note: the hardness measures are present for only 1/3 or samples
chemical <- c("pH_rating","P_rating","K_rating","minor_elements_rating")
biological <- c("organic_matter_rating","ace_soil_protein_index_rating","respiration_rating","active_carbon_rating")
factor_list <- c(physical, biological, chemical, "avg.rating", "tillage")

# Import study
p <- get_study("filtered","full")
  
# Prep metadata
meta <- as.data.frame(as(sample_data(p), "matrix"), stringsAsFactors = F)
meta <- meta[,factor_list]

# Call 'Strip till' as Tillage level 1 and set 4 = 3
if (factor == "tillage"){
  meta$fact[which(meta$fact == "Strip Till")] <- "1"
  meta$fact[which(meta$fact == "4")] <- "3"
}  

for (n in 1:ncol(meta)){
  foo <- as.data.frame(meta[,n], stringsAsFactors = F)
  foo$sampleID <- row.names(foo)
  foo <- foo[complete.cases(foo),]
  foo$factor <- colnames(meta)[n]
  colnames(foo)[1] <- "value"
  foo$value <- as.numeric(foo$value)
  foo$scaled <- scale(foo$value)
  
  if (n == 1){
    results <- foo
  } else {
    results <- rbind(results, foo)
  }
}

# add in grouping
results$group <- NA
results$group[which(results$factor %in% physical)] <- "physical"
results$group[which(results$factor == "tillage")] <- "physical"
results$group[which(results$factor %in% biological)] <- "biological"
results$group[which(results$factor == "root_pathogen_pressure_rating")] <- "biological"
results$group[which(results$factor %in% chemical)] <- "chemical"
results$group[which(results$factor == "avg.rating")] <- "overall.score"
results$group <- factor(results$group, levels = c("overall.score","biological","chemical","physical"))

# Basic density
p <- ggplot(results, aes(x=scaled, color = factor)) + geom_density(adjust = 2.5) + facet_wrap(~group) + theme_bw() +  xlim(-5, 5)
p
ggsave(p, filename='density.distributions.ratings.pdf', height=10, width=16)
