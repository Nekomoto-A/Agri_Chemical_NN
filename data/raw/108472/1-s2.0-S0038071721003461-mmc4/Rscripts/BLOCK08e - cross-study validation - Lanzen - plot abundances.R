library(phyloseq)
library(plyr)
library(reshape2)
library(ggplot2)
library(Hmisc)
library(corrplot)
library(spaa)
source('get_study.R')

# Import feature importance data for Lanzen
OTUs <- readRDS(file = "../models/lanzen.shared.important.features.rds")  # From BLOCK08d

# Import and prune Lanzen microbiome data
p <-  readRDS(file = "../microbiome.data/p_lanzen.final.rds")
p <- subset_taxa(p, taxa_sums(p) > 0)
p <- prune_taxa(OTUs, p)

# Prep data for plotting
counts <- as.data.frame(t(as(otu_table(p), "matrix")), stringsAsFactors = F)
counts$sampleID <- row.names(counts)
counts <- melt(counts)
colnames(counts)[2:3] <- c("OTU","Abundance")
meta <- as.data.frame(as(sample_data(p), "matrix"), stringsAsFactors = F)
meta$sampleID <- row.names(meta)
x <- merge(meta, counts, by = "sampleID")

# Which Lanzen metrics are correlated
meta <- meta[,c("pH","yield.dry.wt","yield.fresh.wt","organic_matter","CO2.respiration","induced.respiration","potassium","compaction","penetrability")]
x.cor <- rcorr(as.matrix(meta[,-10]), type = "pearson")

corrplot(x.cor$r, method = "circle", type = "upper",p.mat = x.cor$P, sig.level = 0.05, insig = c("pch"), pch = 1, pch.cex=1)

# denote correlation direction
x.corr <- x
x.corr <- reshape(x[,c("sampleID","OTU","Abundance","yield.dry.wt")], direction = "wide", idvar = c("sampleID","yield.dry.wt"), timevar = "OTU")
colnames(x.corr) <- gsub("Abundance.","",colnames(x.corr))
row.names(x.corr) <-x.corr$sampleID
x.corr$sampleID <- NULL
x.corr <- rcorr(as.matrix(x.corr), type = "pearson")
x.corr <- x.corr$r
x.corr <- dist2list(as.dist(x.corr))
x.corr <- subset(x.corr, row == "yield.dry.wt")
x.corr$direction <- "negative"
x.corr$direction[which(x.corr$value > 0)] <- "positive"
x.corr <- subset(x.corr, value != 0)
saveRDS(x.corr, file ="data/lanzen.important.features.health.rating.corr.rds")
x <- merge(x, x.corr[,c("col","direction")], by.x = "OTU", by.y = "col")

# Sum all OTU abundance
plot_me <- ddply(x, ~ sampleID + yield.dry.wt, summarise, total = sum(Abundance))

# plot OTU
plot <- ggplot(plot_me, aes(yield.dry.wt, total)) + geom_point() + geom_smooth(method = "lm", se = FALSE)
plot + theme(legend.position="none")


### 
### Permutational testing of correlation strength

# Import and prune Lanzen microbiome data
p <-  readRDS(file = "../microbiome.data/p_lanzen.final.rds")
p <- subset_taxa(p, taxa_sums(p) > 0)

for (n in 1:1000){
  OTUs <- sample(taxa_names(p), length(readRDS(file = "../models/lanzen.shared.important.features.rds")))
  p_foo <- prune_taxa(OTUs, p)  
  
  # Prep data for plotting
  counts <- as.data.frame(t(as(otu_table(p_foo), "matrix")), stringsAsFactors = F)
  counts$sampleID <- row.names(counts)
  counts <- melt(counts)
  colnames(counts)[2:3] <- c("OTU","Abundance")
  meta <- as.data.frame(as(sample_data(p_foo), "matrix"), stringsAsFactors = F)
  meta$sampleID <- row.names(meta)
  x <- merge(meta, counts, by = "sampleID")
  
  # denote correlation direction
  x.corr <- x
  x.corr <- reshape(x[,c("sampleID","OTU","Abundance","yield.dry.wt")], direction = "wide", idvar = c("sampleID","yield.dry.wt"), timevar = "OTU")
  colnames(x.corr) <- gsub("Abundance.","",colnames(x.corr))
  row.names(x.corr) <-x.corr$sampleID
  x.corr$sampleID <- NULL
  x.corr <- rcorr(as.matrix(x.corr), type = "pearson")
  x.corr <- x.corr$r
  x.corr <- dist2list(as.dist(x.corr))
  x.corr <- subset(x.corr, row == "yield.dry.wt")
  x.corr$direction <- "negative"
  x.corr$direction[which(x.corr$value > 0)] <- "positive"
  x.corr <- subset(x.corr, value != 0)
  x.corr$seed <- n
  
  if (n == 1){
    results <- x.corr
  } else {
    results <- rbind(results, x.corr)
  }
}

#saveRDS(results, file = "data/lanzen.permuted.corr.rds")

# random subsets
x.perm <- ddply(readRDS(file = "../models/lanzen.permuted.corr.rds"), ~ seed + direction, summarise, average = mean(value))
hist(subset(x.perm, direction == "positive")$average)
hist(subset(x.perm, direction == "negative")$average)
mean(subset(x.perm, direction == "positive")$average)
mean(subset(x.perm, direction == "negative")$average)

# imporant feature set
x.corr <- ddply(readRDS(file ="../models/lanzen.important.features.health.rating.corr.rds"), ~ direction, summarise, average = mean(value))
x.corr

length(which(x.perm$average > x.corr[2,2]))/1000 # positive correlates
length(which(x.perm$average < x.corr[1,2]))/1000 # negative correlates


###
### Output Table with Info on Important Features

# get taxonomic Info
taxa <- as.data.frame(as(tax_table(p), "matrix"), stringsAsFactors = F)

# get average and max abundance and proportion of presence
SH <- get_study("minimal.norm", "full") 
SH <- subset_taxa(SH, taxa_names(SH) %in% row.names(taxa))

taxa$average <- NA
taxa$max <- NA
taxa$occurrance <- NA
for (n in 1:nrow(taxa)){
  foo <- subset_taxa(SH, taxa_names(SH) == row.names(taxa)[n])
  taxa$max[n] <- round(max(sample_sums(foo)),1)
  taxa$average[n] <- round(mean(sample_sums(foo)),1)
  taxa$occurrance[n] <- round((length(which(sample_sums(foo) > 0))/length(sample_sums(foo)))*100,1)
}

write.csv(taxa, file = "figures/top.overlap.lazen.info.csv", row.names = T, quote = F)
