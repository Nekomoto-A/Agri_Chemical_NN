library(plyr)
library(reshape2)
library(ggplot2)
library(phyloseq)
library(vegan)
library(dendextend)
library(limma)
source("get_study.R")

# Import feature importance data
x <- readRDS(file = "../models/compiled.feature.selection.results.final.rds") # output from "ML.block03.feature.importance.SH.R"

## Work with OTUs
x <- subset(x, rank == "ASV")
x <- x[order(x$ratio),]

### What was the average decrease for classification vs. regression
###
mean(x[grep("category",x$file),"ratio"])
mean(x[grep("rating",x$file),"ratio"])


### How many OTUs contributed at some level to model accuracy in RF data for health ratings
###
foo <- subset(x, dataset == "minimal.norm" & model == "RandomForest")
foo <- foo[-grep("category",foo$health.metric),]
length(unique(subset(foo, ratio < 0.9999)$names))
length(unique(subset(foo, ratio < 0.998)$names))
length(unique(foo$names))


###
### Table S9 - Overlap of important features among health metrics
t9 <- subset(x, dataset == "minimal.norm" & model == "RandomForest" & ratio < 0.998)
t9 <- t9[-grep("category",t9$health.metric),]
t9 <- unique(t9[,c("health.metric","names")])
t9$count <- 1
t9 <- reshape(t9, direction = "wide", idvar = "names", timevar = "health.metric")
t9[is.na(t9)] <- 0
colnames(t9) <- gsub("count.","", colnames(t9))
colnames(t9) <- gsub("_rating","", colnames(t9))

# add in taxonomic classifications
p <- readRDS(file = "../microbiome.data/p_soil.health.minimal.new.tax.rds") # updated taxonomic classification Silva 13.8
p <- prune_taxa(row.names(t9), p)
tax_table <- as.data.frame(as(tax_table(p), "matrix"), stringsAsFactors = F)
tax_table$OTU <- row.names(tax_table)
t9 <- merge(t9, tax_table, by.x = "names", by.y = "OTU")

# Order and output
row.names(t9) <- t9$names
t9$names <- NULL
t9 <- t9[rev(order(rowSums(t9[,1:15]))),]

write.csv(t9, file = "figures/tableS9.csv", quote = F, row.names = T)


###
### Plot top 20 important features for predicting health rating

# subset to best rating models
x.health <- subset(x, health.metric == "avg.rating" & dataset == "minimal.norm" & model == "RandomForest")

# Grab top 20 important features
foo <- ddply(x.health, ~ names, summarise, avg = mean(ratio))
foo <- foo[order(foo$avg),]
x.health <- subset(x.health, names %in% foo[1:20,"names"])
x.health <- x.health[order(x.health$ratio),]
x.health$names <- factor(x.health$names, levels = rev(foo[1:20,"names"]))

# Plot
plot <- ggplot(x.health, aes(x=names, y=ratio)) +  
  geom_boxplot() +  
  ylab("Proportional decrease") +  
  coord_flip() 
plot
ggsave(plot, file = "figures/feature.importance.health.rating.plot.pdf", height=5, width=10)


###
### Quantify overlap among health metrics

# average all importance ratios
foo <- subset(x, dataset == "minimal.norm" & model == "RandomForest" & ratio < 0.99999 & health.metric != "health.category")
x.health <- ddply(foo, ~ health.metric + names, summarise, avg.ratio = mean(ratio))

# filter most impactful
x.health <- subset(x.health, avg.ratio < 0.998)
x.health$avg.ratio <- NULL

# make count table
x.health$count <- 1
x.health <- reshape(x.health, direction = "wide", idvar = "health.metric", timevar = "names")
x.health[is.na(x.health)] <- 0
colnames(x.health) <- gsub("count.","",colnames(x.health))
row.names(x.health) <- x.health$health.metric
x.health$health.metric <- NULL


## 
## Plot hierarchical clustering

# get distance matrix
x.dist <- vegdist(x.health, method = "bray")

# do hierarchical clustering
tree <- hclust(x.dist, method = "complete")
plot(as.dendrogram(tree))

## 
## How many overlap within physical, chemical and biological
physical <- c("water_capacity_rating", "surface_hardness_rating", "subsurface_hardness_rating", "aggregate_stability_rating")  # Note: the hardness measures are present for only 1/3 or samples
chemical <- c("pH_rating","P_rating","K_rating","minor_elements_rating")
biological <- c("organic_matter_rating","ace_soil_protein_index_rating","respiration_rating","active_carbon_rating")
other <- c("soil_texture_clay","soil_texture_sand","soil_texture_silt")

x.health$metric <- row.names(x.health)
x.health <- melt(x.health)
x.health$class <- NA
x.health$class[which(x.health$metric %in% physical)] <- "physical"
x.health$class[which(x.health$metric %in% chemical)] <- "chemical"
x.health$class[which(x.health$metric %in% biological)] <- "biological"
x.health$class[which(x.health$metric %in% other)] <- "other"
x.health <- subset(x.health, !(is.na(class)))
foo <- ddply(x.health, ~ class + variable, summarise, total = sum(value))
table(foo$class, foo$total)
subset(foo, total == 3 & class == "biological")

#27+7/(126+27+7) # 21% 
#0/(158) 
#29+6/(255+29+6) # 12%

foo <- ddply(x.health, ~ variable, summarise, total = sum(value))
foo <- foo[rev(order(foo$total)),]
nrow(subset(foo, total > 1))/nrow(foo)


###
### Quantify the agreement in important features between health category and rating

top.cat <- as.character(unique(subset(x, health.metric == "health.category" & model == "RandomForest" & dataset == "minimal.norm" & ratio < 0.998)$names))
top.rat <- as.character(unique(subset(x, health.metric == "avg.rating" & model == "RandomForest" & dataset == "minimal.norm" & ratio < 0.998)$names))

# Draw a Venn Diagram
venn_bin <- data.frame(OTUs = unique(c(top.cat, top.rat)))
venn_bin[,"classification"] <- venn_bin$OTUs %in% top.cat
venn_bin[,"regression"] <- venn_bin$OTUs %in% top.rat
venn_counts <- vennCounts(venn_bin[,c("classification","regression")])
pdf("figures/venn.important.features.modes.pdf", width=8,height=6)
vennDiagram(venn_counts, names = c("classification","regression"), cex = 1, counts.col = "red")  
dev.off()

###
### Quantify overlap between ACE Protein, Active Carbon and Health rating
top.rat <- as.character(unique(subset(x, health.metric == "avg.rating" & model == "RandomForest" & dataset == "minimal.norm" & ratio < 0.998)$names))
top.ace <- as.character(unique(subset(x, health.metric == "ace_soil_protein_index_rating" & model == "RandomForest" & dataset == "minimal.norm" & ratio < 0.998)$names))
top.carb <- as.character(unique(subset(x, health.metric ==  "active_carbon_rating" & model == "RandomForest" & dataset == "minimal.norm" & ratio < 0.998)$names))

# Draw a Venn Diagram
venn_bin <- data.frame(OTUs = unique(c(top.ace, top.carb, top.rat)))
venn_bin[,"health rating"] <- venn_bin$OTUs %in% top.rat
venn_bin[,"ACE protein"] <- venn_bin$OTUs %in% top.ace
venn_bin[,"active carbon"] <- venn_bin$OTUs %in% top.carb
venn_counts <- vennCounts(venn_bin[,c("health rating","ACE protein","active carbon")])
pdf("figures/venn.important.features.sand.vs.health.pdf", width=8,height=6)
vennDiagram(venn_counts, names = c("health rating","ACE protein","active carbon"), cex = 1, counts.col = "red")  
dev.off()


### 
### What proportion of samples are important features found in vs. non-important features

# Import feature importance data
x <- readRDS(file = "data/compiled.feature.selection.results.final.rds")
x <- subset(x, rank == "ASV" & dataset == "minimal.norm" & model == "RandomForest")
x <- x[order(x$ratio),]
x <- x[-grep("category",x$health.metric),]

important <- as.character(unique(subset(x,  ratio < 0.998)$names))
not <- as.character(setdiff(unique(x$names), important))

# calculate average abundances
p <- get_study("minimal.norm", "full")   
p <- prune_samples(readRDS(file = "data/ML.sample.set.rds"), p)
p <- subset_taxa(p, taxa_sums(p) > 0)
p <- subset_taxa(p, taxa_names(p) %in% unique(x$names))


# get average and max abundance and proportion of presence
taxa <- as.data.frame(as(tax_table(p), "matrix"), stringsAsFactors = F)
taxa$average <- NA
taxa$max <- NA
taxa$occurrance <- NA

for (n in 1:nrow(taxa)){
  foo <- subset_taxa(p, taxa_names(p) == row.names(taxa)[n])
  taxa$max[n] <- round(max(sample_sums(foo)),1)
  taxa$average[n] <- round(mean(sample_sums(foo)),1)
  taxa$occurrance[n] <- round((length(which(sample_sums(foo) > 0))/length(sample_sums(foo)))*100,1)
}

taxa$important <- 0
taxa$important[which(row.names(taxa) %in% important)] <- 1
saveRDS(taxa, file = "data/abundance.of.important.features.rds")

# Average
#ddply(taxa, ~ important, summarise, avg = mean(average))
ddply(taxa, ~ important, summarise, occ = mean(occurrance))

# Max
ddply(taxa, ~ important, summarise, occ = max(occurrance))

# Min
ddply(taxa, ~ important, summarise, occ = min(occurrance))
