library(phyloseq)
library(plyr)
library(dplyr)
library(ggplot2)
library(vegan)
library(spaa)
library(Rtsne)
library(viridis)
source("get_study.R")

####
## User Input
dataset <- "filter.css"

# define biological, chemical and physical health metrics 
physical <- c("water_capacity_rating", "surface_hardness_rating", "subsurface_hardness_rating", "aggregate_stability_rating")  # Note: the hardness measures are present for only 1/3 or samples
chemical <- c("pH_rating","P_rating","K_rating","minor_elements_rating")
biological <- c("organic_matter_rating","ace_soil_protein_index_rating","respiration_rating","active_carbon_rating")

##############
## Import data
p <- get_study(dataset, "full")

# Subset to downsampled dataset (capped at 10 samples per geo_group and management)
p <- subset_samples(p, sample_names(p) %in% readRDS(file = "data/ML.sample.set.rds"))
p <- subset_taxa(p, taxa_sums(p) > 0)

###################
## t-SNE Ordination
set.seed(33)

# 260 colours for use in geo_groups plot
colour_palette <- c("#000000","#FFFF00","#1CE6FF","#FF34FF","#FF4A46","#008941","#006FA6","#A30059","#FFDBE5","#7A4900","#0000A6","#63FFAC","#B79762","#004D43","#8FB0FF","#997D87","#5A0007","#809693","#FEFFE6","#1B4400","#4FC601","#3B5DFF","#4A3B53","#FF2F80","#61615A","#BA0900","#6B7900","#00C2A0","#FFAA92","#FF90C9","#B903AA","#D16100","#DDEFFF","#000035","#7B4F4B","#A1C299","#300018","#0AA6D8","#013349","#00846F","#372101","#FFB500","#C2FFED","#A079BF","#CC0744","#C0B9B2","#C2FF99","#001E09","#00489C","#6F0062","#0CBD66","#EEC3FF","#456D75","#B77B68","#7A87A1","#788D66","#885578","#FAD09F","#FF8A9A","#D157A0","#BEC459","#456648","#0086ED","#886F4C","#34362D","#B4A8BD","#00A6AA","#452C2C","#636375","#A3C8C9","#FF913F","#938A81","#575329","#00FECF","#B05B6F","#8CD0FF","#3B9700","#04F757","#C8A1A1","#1E6E00","#7900D7","#A77500","#6367A9","#A05837","#6B002C","#772600","#D790FF","#9B9700","#549E79","#FFF69F","#201625","#72418F","#BC23FF","#99ADC0","#3A2465","#922329","#5B4534","#FDE8DC","#404E55","#0089A3","#CB7E98","#A4E804","#324E72","#6A3A4C","#83AB58","#001C1E","#D1F7CE","#004B28","#C8D0F6","#A3A489","#806C66","#222800","#BF5650","#E83000","#66796D","#DA007C","#FF1A59","#8ADBB4","#1E0200","#5B4E51","#C895C5","#320033","#FF6832","#66E1D3","#CFCDAC","#D0AC94","#7ED379","#012C58","#7A7BFF","#D68E01","#353339","#78AFA1","#FEB2C6","#75797C","#837393","#943A4D","#B5F4FF","#D2DCD5","#9556BD","#6A714A","#001325","#02525F","#0AA3F7","#E98176","#DBD5DD","#5EBCD1","#3D4F44","#7E6405","#02684E","#962B75","#8D8546","#9695C5","#E773CE","#D86A78","#3E89BE","#CA834E","#518A87","#5B113C","#55813B","#E704C4","#00005F","#A97399","#4B8160","#59738A","#FF5DA7","#F7C9BF","#643127","#513A01","#6B94AA","#51A058","#A45B02","#1D1702","#E20027","#E7AB63","#4C6001","#9C6966","#64547B","#97979E","#006A66","#391406","#F4D749","#0045D2","#006C31","#DDB6D0","#7C6571","#9FB2A4","#00D891","#15A08A","#BC65E9","#FFFFFE","#C6DC99","#203B3C","#671190","#6B3A64","#F5E1FF","#FFA0F2","#CCAA35","#374527","#8BB400","#797868","#C6005A","#3B000A","#C86240","#29607C","#402334","#7D5A44","#CCB87C","#B88183","#AA5199","#B5D6C3","#A38469","#9F94F0","#A74571","#B894A6","#71BB8C","#00B433","#789EC9","#6D80BA","#953F00","#5EFF03","#E4FFFC","#1BE177","#BCB1E5","#76912F","#003109","#0060CD","#D20096","#895563","#29201D","#5B3213","#A76F42","#89412E","#1A3A2A","#494B5A","#A88C85","#F4ABAA","#A3F3AB","#00C6C8","#EA8B66","#958A9F","#BDC9D2","#9FA064","#BE4700","#658188","#83A485","#453C23","#47675D","#3A3F00","#061203","#DFFB71","#868E7E","#98D058","#6C8F7D","#D7BFC2","#3C3E6E","#D83D66","#2F5D9B","#6C5E46","#D25B88","#5B656C","#00B57F","#545C46","#866097","#365D25","#252F99","#00CCFF","#674E60","#FC009C","#92896B")

# Filter to size where tSNE doesn't give memory error...
threshold = 40
p <- subset_taxa(p, taxa_sums(p) > threshold)

# Prep Data for tSNE
counts <- as.data.frame(as(t(otu_table(p)), "matrix"))
design <- as.data.frame(as(sample_data(p), "matrix"), stringsAsFactors = F)

# Run tSNE
tsne = Rtsne(counts, dims = 2, perplexity = 30, verbose = TRUE)
saveRDS(tsne, file = "data/tsne.rds")

## Plot Factors of Interest
tsne <- readRDS(file = "data/tsne.rds")
plot_me <- cbind(as.data.frame(tsne["Y"]), design)
plot_me$avg.rating <- as.numeric(plot_me$avg.rating)
plot1 <- ggplot(plot_me, aes(Y.1, Y.2, colour = manage_group)) + geom_point() + scale_color_manual(values=colour_palette[1:length(unique(plot_me$manage_group))]) + theme(legend.position = "none") 
plot2 <- ggplot(plot_me, aes(Y.1, Y.2)) + geom_point(aes(color = avg.rating)) + scale_color_viridis(option = "D")

ggsave(plot1, filename=paste("figures/tSNE",dataset,'management_group.pdf',sep="."), height=8, width=10)
ggsave(plot2, filename=paste("figures/tSNE",dataset,'soil.health.rating.pdf',sep="."), height=8, width=10)


############
## PERMANOVA

# aim 1: determine contributions of each soil health metric to MCC
# aim 2: determine contributions for biological, chemical and physical to MCC

with_geo <- "Y"  # "Y" | "N"
model_parameters <- "all"  # "main", "all" | "biological" | "chemical" | "physical" | "biological_chemical" | "biological_physical" | "physical_chemical"
categorical <- "N" # "Y" | "N"

if (model_parameters == "all"){
  factors <- c(biological, chemical, physical)
} else if (model_parameters == "biological_chemical"){
  factors <- c(biological, chemical)
} else if (model_parameters == "physical_chemical"){
  factors <- c(chemical, physical)
} else if (model_parameters == "biological_physical"){
  factors <- c(biological, physical)
} else if (model_parameters == "main"){
  factors <- c("tillage","geo_group","health.category","soil_texture_class")
} else {
  factors <- get(model_parameters)
}

# Prepare Design Matrix
design <- as.data.frame(as(sample_data(p), "matrix"), stringsAsFactors = F)
design$geo_group <- trimws(design$geo_group)

# Select factors for model
if (with_geo == "Y"){
  factors <- append(factors, "geo_group")
} 

if (categorical == "Y" & model_parameters != "main"){
  factors <- gsub("_rating",".category", factors)
}

if (model_parameters == "all" & categorical == "N"){
  factors <- append(factors, c("soil_texture_clay","soil_texture_sand","soil_texture_silt"))
}
design <- design[,factors]

remove <- c("surface_hardness_rating","subsurface_hardness_rating","surface_hardness.category","subsurface_hardness.category")
if (any(remove %in% colnames(design))){
  design[,remove] <- list(NULL)
}

# remove any samples with missing values
to_remove <- setdiff(rownames(design), rownames(na.omit(design)))
foo <- subset_samples(p, !(sample_names(p) %in% to_remove))
design <- na.omit(design)

samples <- row.names(design)
if (categorical == "N" & model_parameters != "main"){
  design <- data.frame(apply(design, 2, as.numeric))
} else {
  design <- data.frame(apply(design, 2, as.factor))
}
rownames(design) <- samples

if (with_geo == "Y" & model_parameters != "main"){
  design$geo_group <- as.factor(design$geo_group)
}

# Make Count Matrix
counts <- as.data.frame(t(as(otu_table(foo), "matrix")), stringsAsFactors = F)

# Set permutations
permutations <- 999
perm <- how(nperm = permutations)

# Calculate Distance (pref: weighted Unifrac)
x.dist <- try(readRDS(paste("data/",dataset,".Unifrac.dist.perm.",categorical,".",model_parameters,".rds", sep="")), silent = TRUE)

if (class(x.dist) == "try-error") {
  x.dist <- UniFrac(prune_samples(row.names(design), p), weighted=TRUE, normalized=TRUE, parallel=FALSE)
  saveRDS(x.dist, file = paste("data/",dataset,".Unifrac.dist.perm.",categorical,".",model_parameters,".rds", sep=""))
}

# Sanity check
length(labels(x.dist)) == nrow(design)

# Run PERMANOVA
if (with_geo == "Y"){
  permanova <- adonis2(x.dist ~ ., data = design, strata = "geo_group", permutations = 9999) 
} else {
  permanova <- adonis2(x.dist ~ ., data = design, permutations = perm)
}

permanova


#################################
## Within Group Unifrac Distances
library(pgirmess)
citation('pgirmess')

# Calculate weighted Unifrac
x.dist <- try(readRDS(paste("data/",dataset,".Unifrac.dist.rds", sep="")), silent = TRUE)

if (class(x.dist) == "try-error") {
  x.dist <- UniFrac(p, weighted=TRUE, normalized=TRUE, parallel=FALSE)
  saveRDS(x.dist, file = paste("data/",dataset,".Unifrac.dist.rds", sep=""))
}

# Prepare soil health data
x <- dist2list(x.dist)
x <- subset(x, value != 0)
colnames(x)[1] <- "sampleID"
design <- as.data.frame(as(sample_data(p), "matrix"), stringsAsFactors = F)
design$sampleID <- row.names(design)
x <- merge(x, design[,c("sampleID","health.category")], by = "sampleID")
colnames(x)[1] <- "sample1"
colnames(x)[2] <- "sampleID"
colnames(x)[4] <- "sample1.cat"
x <- merge(x, design[,c("sampleID","health.category")], by = "sampleID")
colnames(x)[1] <- "sample2"
colnames(x)[5] <- "sample2.cat"

# Remove duplicates
x <- x[!duplicated(apply(x[1:2], 1, function(x) toString(sort(x)))),]

# Within dist
x$cat <- NA
x$cat[which(x$sample1.cat == "(20,40]" & x$sample2.cat == "(20,40]")] <- "(20,40]"
x$cat[which(x$sample1.cat == "(40,60]" & x$sample2.cat == "(40,60]")] <- "(40,60]"
x$cat[which(x$sample1.cat == "(60,80]" & x$sample2.cat == "(60,80]")] <- "(60,80]"
x$cat[which(x$sample1.cat == "(80,100]" & x$sample2.cat == "(80,100]")] <- "(80,100]"
x <- subset(x, !(is.na(x$cat)))

x.rarefy <- rbind(subset(x, cat == "(20,40]"), sample_n(subset(x, cat == "(40,60]"), min(table(x$cat)), replace = F))
x.rarefy <- rbind(x.rarefy, sample_n(subset(x, cat == "(60,80]"), min(table(x$cat)), replace = F))
x.rarefy <- rbind(x.rarefy, sample_n(subset(x, cat == "(80,100]"), min(table(x$cat)), replace = F))

plot <- ggplot(x.rarefy, aes(x=cat, y=value, colour = cat)) + geom_boxplot(outlier.shape = NA) + ylab("Unifrac distance") 
plot <- plot + geom_jitter(width=0.1, alpha=0.5) # + coord_flip()
plot
ggsave(plot, filename='within.category.unifrac.dist.pdf', height=16, width=16)

# Note: Unifrac represents the proportion of *unshared* branch length over total branch length.
#       Thus, the higher the value, the greater the dissimilarity, with 1 = no shared branches.

# Data is not normal
#shapiro.test(subset(x.rarefy, cat == "(20,40]")$value)
#shapiro.test(subset(x.rarefy, cat == "(40,60]")$value)
#shapiro.test(subset(x.rarefy, cat == "(60,80]")$value)
#shapiro.test(subset(x.rarefy, cat == "(80,100]")$value)

# Multi-comparison Kruskal-Wallis Test
kruskalmc(value ~ cat, data = x.rarefy)


#######################################################################
## Mantel test of covariance in Unifrac distance and Soil health rating

physical <- c("water_capacity_rating", "surface_hardness_rating", "subsurface_hardness_rating", "aggregate_stability_rating")  # Note: the hardness measures are present for only 1/3 or samples
chemical <- c("pH_rating","P_rating","K_rating","minor_elements_rating")
biological <- c("organic_matter_rating","ace_soil_protein_index_rating","respiration_rating","active_carbon_rating")

# Import dataset
p <- get_study("filter.css", "full")

# Subset to downsampled dataset (capped at 10 samples per geo_group and management)
p <- subset_samples(p, sample_names(p) %in% readRDS(file = "data/ML.sample.set.rds"))

# create distance matrix based on microbial community composition
microbiome.dist <- try(readRDS(paste("data/filter.css.Unifrac.dist.rds", sep="")), silent = TRUE)
if (class(microbiome.dist) == "try-error") {
  microbiome.dist <- UniFrac(p, weighted=TRUE, normalized=TRUE, parallel=FALSE)
  saveRDS(microbiome.dist, file = paste("data/",dataset,".Unifrac.dist.rds", sep=""))
}

# create distance matrix based on soil health metrics
all.metrics <- as.data.frame(as(sample_data(p), "matrix"), stringsAsFactors = F)
physical.special <- c("water_capacity_rating", "aggregate_stability_rating")  # Hardness measures removed

# run through all combinations
sets <- c("all","physical.special","chemical","biological")
combos <- combn(c("physical.special","chemical","biological"), 2)
for (n in 1:ncol(combos)){
  sets<-append(sets, paste(combos[,n], collapse="_"))
}

count <- 1
for (health_set in sets){
  if (health_set == "all"){
    health.metrics <- all.metrics[,grep("_rating", colnames(all.metrics))]
    health.metrics[,c("surface_hardness_rating","subsurface_hardness_rating","root_pathogen_pressure_rating")]  <- list(NULL) # These metrics are too incomplete
  } else {
    metrics <- unlist(strsplit(health_set, "_"))    
    if (length(metrics) > 1){
      metrics <- c(get(metrics[1]),get(metrics[2]))
    } else {
      metrics <- get(metrics[1])
    }
    health.metrics <- all.metrics[,metrics]
  } 

  # convert to numeric and calculate distance
  health.metrics <- mutate_all(health.metrics, function(x) as.numeric(as.character(x)))  
  health.dist <- vegdist(health.metrics, method="bray", binary=F, na.rm = T)

  # Perform mantel
  output <- mantel(health.dist, microbiome.dist, method = "pearson", permutations = 999, na.rm = TRUE)
  
  if (count == 1){
    results <- data.frame(metric = health_set, correlation = output$statistic, p_value = output$signif, permutations = output$permutations, stringsAsFactors = F)
    count <- count + 1
  } else {
    results <- rbind(results, data.frame(metric = health_set, correlation = output$statistic, p_value = output$signif, permutations = output$permutations, stringsAsFactors = F))
  }
}

saveRDS(results, file = "data/mantel.test.results.rds")
