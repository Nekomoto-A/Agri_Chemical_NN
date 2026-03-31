library(phyloseq)
library(plyr)
library(ggplot2)
source("get_study.R")

from_basic = "N"   # "Y" | "N"   (only chose yes if you're running on a server with high memory)

##################################
##  Calculate Diversity & Richness
set.seed(7)

count = 1
if (from_basic == "Y"){


  permutations = 200
  indices = c("Observed", "Chao1", "ACE", "Shannon", "Simpson", "InvSimpson")
  
  # Import data  
  p <- get_study("filtered","full")
  
  # Subtract 1 from all counts since DADA2 tosses singletons
  otu_table(p) <- otu_table(p)-1
  otu_table(p)[otu_table(p) < 0] <- 0
  p <- subset_taxa(p, taxa_sums(p) > 0)
  
  for (i in 1:permutations){
    foo <- rarefy_even_depth(p)
    
    foo <- plot_richness(foo, c("health.category"), measures=indices)  #Note: it doesn't matter that "treatment" is specified. All sample data is preserved
    
    if (count == 1){
      x.data <- foo$data
      x.data$perm <- i
    } else {
      foodata <- foo$data
      foodata$perm <- i
      x.data <- rbind(x.data, foodata)
    }
    count = count + 1
    print(count)
  }
  
  colnames(x.data)[grep("variable",colnames(x.data))] <- "diversity.metric"
  colnames(x.data)[grep("value",colnames(x.data))] <- "estimate"
  colnames(x.data)[grep("samples",colnames(x.data))] <- "sampleID"
  
  saveRDS(x.data, file = "data/soil.health.alpha-diversity.rds")
} else {
  x <- readRDS(file = "data/soil.health.alpha-diversity.rds")
}

## Add in Pielou's Evenness Metric (done after the main calculation run)
if (from_basic == "Y"){
  
  #Import data
  x <- readRDS(file = "data/soil.health.alpha-diversity.rds")
  
  # Grab data needed for calculation
  observed <- subset(x, diversity.metric == "Observed")
  shannon <- subset(x, diversity.metric == "Shannon")
  
  #any(shannon$samples != observed$samples)  #sanity check
  
  # Calculate Evenness
  J <- shannon$estimate/log(observed$estimate)  # Pielou's evenness
  pielou <- shannon
  pielou$diversity.metric <- "Pielou"
  pielou$estimate <- J
  pielou$se <- NA
  
  # Add back to main dataframe  
  x <- rbind(x, pielou)
  
  saveRDS(x, file = "data/soil.health.alpha-diversity.rds")
}


########
# Plot 

#### USER INPUT : choose diversity metric
metric <- "Shannon" # options include : "Observed", "Chao1", "ACE", "Shannon", "Simpson", "InvSimpson", "Pielou"

### Choose Soil Health Variable to use Display
###
physical <- c("water_capacity.category", "surface_hardness.category", "subsurface_hardness.category", "aggregate_stability.category")  # Note: the hardness measures are present for only 1/3 or samples
chemical <- c("pH.category","P.category","K.category","minor_elements.category")
biological <- c("organic_matter.category","ace_soil_protein_index.category","respiration.category","active_carbon.category")
factor_list <- c("tillage", "health.category", physical, chemical, biological)

# subset to downsampled dataset (capped at 10 samples per geo_group and management)
x <- subset(x, sampleID %in% readRDS(file = "data/ML.sample.set.rds"))

fact <- "health.category"

# Plot each factor
count = 1
for (fact in factor_list){
  
  # Average all estimates across random rarefication 
  plot.data <- ddply(subset(x, diversity.metric == metric), ~ sampleID + get(fact) + diversity.metric, summarise, average = mean(estimate))
  colnames(plot.data)[2] <- "factor"
  
  if (fact == "tillage"){
    plot.data$factor <- gsub("4","3", plot.data$factor)
    plot.data <- subset(plot.data, !(is.na(factor)))
    
    # Call 'Strip till' as Tillage level 1
    plot.data$factor[which(plot.data$factor == "Strip Till")] <- "2"
  }
  
  # Plot
  plot <- ggplot(plot.data, aes(factor, average)) + ggtitle(paste(metric,fact, sep=" -- ")) + geom_boxplot() + ylab(paste(metric)) + xlab(fact)  + theme_bw()
  plot <- plot + geom_jitter(width = 0.1, alpha=0.5)
  print(plot)
  
  ### TukeyHSD Pairwise Comparison and Interactions
  fit <- lm(average ~ factor, plot.data)
  fit <- aov(fit)
  
  averages <- data.frame(ddply(plot.data, ~ factor, summarise, average = mean(average)))
  averages$measure <- fact
  stats <- data.frame(TukeyHSD(fit)$factor)
  stats$measure <- fact
  
  if (count == 1){
    r1 <- averages
    r2 <- stats
    count = count + 1
  } else {
    r1 <- rbind(r1, averages)
    r2 <- rbind(r2, stats)
  }
   
  writeLines("Save plots?")
  switch(menu(c("Discard","Continue","Exit")), step1 <- "Discard", step1 <- "Save to File", step1 <- "Exit")
  
  if (step1 == "Exit"){
    break
  } else if (step1 == "Save to File"){
    ggsave(plot, filename=paste(metric,fact,"alpha-diversity.boxplot.pdf",sep="."), width= 11, height=8.5)
  }
}

write.csv(r1, file = paste("average.",metric,".diversity.csv",sep=""))
write.csv(r2, file = paste("stats.",metric,".diversity.csv",sep=""))


# Check Balance of Classes for Each Soil Health Metric
library(reshape)
count = 1
for (fact in factor_list){
  
  # Average all estimates across random rarefication 
  plot.data <- ddply(subset(x, diversity.metric == metric), ~ sampleID + get(fact) + diversity.metric, summarise, average = mean(estimate))
  colnames(plot.data)[2] <- "factor"
  
  if (fact == "tillage"){
    plot.data$factor <- gsub("4","3", plot.data$factor)
    plot.data <- subset(plot.data, !(is.na(factor)))
    
    # Call 'Strip till' as Tillage level 1
    plot.data$factor[which(plot.data$factor == "Strip Till")] <- "2"
  }
  
  # Plot
  if (count == 1){
    results <- data.frame(table(plot.data[,"factor"]), factor = fact)
    count <- count + 1
  } else {
    results <- rbind(results, data.frame(table(plot.data[,"factor"]), factor = fact))
  }
}

results <- reshape(results, direction = "wide", idvar = "factor", timevar = "Var1")
write.csv(results, file = "samples.per.health.metric.class.csv", row.names = F, quote = F)
