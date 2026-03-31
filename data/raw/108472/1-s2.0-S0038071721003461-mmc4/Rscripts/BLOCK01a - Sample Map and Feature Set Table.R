library(ggmap)
library(plyr)
library(phyloseq)

########################
## Figure 1 - Sample Map

# Import Data
p <- readRDS(file = "data/p_SSU.soil.health.filter.norm.rds")
p <- prune_samples(readRDS(file = "data/ML.sample.set.rds"), p)
x <- as.data.frame(as(sample_data(p), "matrix"), stringsAsFactors = F)

# Prep Data
x$Latitude <- as.numeric(x$Latitude)
x$Longitude <- as.numeric(x$Longitude)
x$count <- 1
plot_me <- ddply(x, ~ manage_group, summarise, total = sum(count))
plot_me$manage_group <- as.factor(plot_me$manage_group)
x <- x[-which(duplicated(x$manage_group)),]
plot_me <- merge(plot_me, x[,c("manage_group","Latitude","Longitude")], by = "manage_group")

# Mean and median
mean(plot_me$total)
median(plot_me$total)

# Create Map
mp <- NULL
usa <- borders("usa", colour="gray50", fill="gray50") # create a layer of borders
mp <- ggplot() + usa

# Black and White
mp_wo_colour <- mp + geom_point(data=plot_me, aes(x=Longitude, y=Latitude, size = total, alpha = 0.05)) + scale_size_continuous(breaks = c(1,2,4,6,8,10))
mp_wo_colour

# Save
ggsave(mp_wo_colour, filename = "sample.map.manage.groups.pdf", height=8, width=16)


##############################
## Table SX - Feature Set Size
library(phyloseq)
source('get_study.R')
source('get_melt.R')

# Import Data
count <- 1
for (feature_type in c("ASV","Order","Family","Genus")){
  for (dataset in c("minimal.rare", "filter.rare", "filter.css", "filter.norm")){
    if (feature_type == "ASV"){
      p <- get_study(dataset, "full")   
      p <- prune_samples(readRDS(file = "data/ML.sample.set.rds"), p)
      
      # Discard Features (ASV | Taxa) that occur in fewer than 10 samples
      present_absent<-otu_table(p) 
      present_absent[present_absent > 0] <- 1
      p <- prune_taxa(rownames(present_absent)[which(rowSums(present_absent) >= 10)], p)
      
      # Get rid of ASV with zero counts
      p <- subset_taxa(p, taxa_sums(p) > 0)
      
      if (count == 1){
        results <- data.frame(set = dataset, feature = feature_type, total = length(taxa_names(p)))
        count <- count + 1
      } else {
        results <- rbind(results, data.frame(set = dataset, feature = feature_type, total = length(taxa_names(p))))
      }
      
    } else {
      
      # Read in pre-formatted taxonomic counts
      counts <- get_melt(dataset, feature_type, "full")
      
      # Filter out samples removed from design matrix
      counts <- subset(counts, sample %in% readRDS(file = "data/ML.sample.set.rds"))
      
      # Discard Features (ASV | Taxa) that occur in fewer than 10 samples
      present_absent<-counts[,2:ncol(counts)]
      present_absent[present_absent > 0] <- 1
      counts <- counts[,-which(colnames(counts) %in% names(which(colSums(present_absent) < 10)))]
      
      # Remove Taxa with zero counts
      if (any(colSums(counts[,2:ncol(counts)]) == 0)){
        counts <- counts[,-(which(colSums(counts[,2:ncol(counts)]) == 0)+1)]
      }
      
      tot <- length(paste("Taxa",seq(1, ncol(counts)-1, by = 1),sep="_"))
      
      if (count == 1){
        results <- data.frame(set = dataset, feature = feature_type, total = tot)
        count <- count + 1
      } else {
        results <- rbind(results, data.frame(set = dataset, feature = feature_type, total = tot))
      }
    }
  }
}

write.csv(results, file = "feature.set.totals.csv", row.names = F, quote = F)
