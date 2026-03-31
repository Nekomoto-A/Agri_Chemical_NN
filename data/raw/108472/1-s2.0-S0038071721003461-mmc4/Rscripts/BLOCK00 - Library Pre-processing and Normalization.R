library(phyloseq)
library(plyr)
library(reshape2)
library(geosphere)
require(sp)
source("get_dataset.R")
source("CSS_normalization_code.R")

## Custom function to normalize to counts per thousand for each library
normalize_to_cpt <- function(p){
  return(apply(otu_table(p), 2, function(x) x/(sum(x)/1000)))
}

## Function to assign samples a groups based on distance (ie. proximity)
geo_group <- function(p, proximity){

  # subset data from phyloseq  
  data <- as.data.frame(as(sample_data(p)[,c("Longitude","Latitude")], "matrix"))

  # Transform Lat/Long into Spatial Data Structure
  xy <- SpatialPointsDataFrame(matrix(c(data$Longitude,data$Latitude), ncol=2), data.frame(ID=row.names(data)),
                               proj4string=CRS("+proj=longlat +ellps=WGS84 +datum=WGS84"))
  
  # Convert Lat/Long to Geographic distances
  mdist <- distm(xy)
  
  # cluster all points using a hierarchical clustering approach
  hc <- hclust(as.dist(mdist), method="complete")
  
  # Distance with a distance threshold  
  cut <- cutree(hc, h=proximity) 
  
  # Add groups to sample data
  sample_data(p)[,"geo_group"] <- trimws(cut)
  
  return(p)
}

## Re-factor all identical IDs as numbers
refactor <- function(x, factor){
  x[,factor] <- as.factor(x[,factor])
  x[,factor] <- factor(x[,factor], levels = unique(x[,factor]))
  levels(x[,factor]) <- 1:length(levels(x[,factor]))
  
  return(as.numeric(x[,factor]))
}

###
### Datasets Produced
###

# i) original: untouched
# ii) minimal: OTUs present in negative controls removed (no other filtering)
# iii) minimal+rarefy: same as above but rarefied
# iv) minimal+proportioned: same as above but normalized
# v) filtered: 
           # - Minimum sample total set to 6000 reads
           # - OTUs removed with less than a total of 0.0075% of the mean library size (~2.65 counts) across all 550 samples
           # - Chloroplasts and mitochondria filtered
# vi) filtered+rarefy: same as above but rarefied
# vii) normalized: Counts are scaled to library total and presented as counts per thousand reads
# viii) css: cumulative sum scaling

# User input on defining geo_groups
proximity <- 1000 # 'proximity' is the minimum distance (m) to be assigned to a group
    
#######################################
## Custom processing for the following:

# 1) Multiple sequencing runs (done in two separate batched: run1,2,3 then run4,5)
# 2) Misisng and incorrect lat and long information
# 3) Labelling contaminants (i.e. those ASVs present in blanks)
# 4) Calculating categorical values for soil health metrics
# 5) Calculating geogroup based on lat. long. info (i.e. aggregating based on proximity of samples)

# Import Sequencing Runs 1-3
p <- readRDS(file = "../microbiome.data/p_SSU.soil.health.final.rds") # original name: p_SSU.soil.health.silva.june.2019.rds

# Remove Unnecessary columns in sample datasheet
sample_data(p) <- sample_data(p)[,c(3,4,19:26,33:75)]  

# Remove tree (will have to add back in later)
p_foo <- merge_phyloseq(otu_table(p), tax_table(p))
p_foo <- merge_phyloseq(p_foo, sample_data(p))

# Import Sequencing Runs 4 & 5
p4 <- readRDS(file = "../microbiome.data/p_run4.final.rds")
p5 <- readRDS(file = "../microbiome.data/p_run5.final.rds")

# Clean up Runs 4 & 5
# Remove tree and merge both
NTC_4 <- subset_samples(p4, sample_names(p4) == "soil.health.run4_NTC_1")
p4 <- subset_samples(p4, experiment == "soil.health" | experiment == "soil.health.aurora")
p4 <- merge_phyloseq(p4, NTC_4) # Add back in NTC control
p4_foo <- merge_phyloseq(otu_table(p4), tax_table(p4))
p4_foo <- merge_phyloseq(p4_foo, sample_data(p4))

NTC_5 <- subset_samples(p5, sample_names(p5) == "soil.health.run5_NTC_2")
p5 <- subset_samples(p5, experiment == "soil.health" | experiment == "soil.health.aurora")
p5 <- merge_phyloseq(p5, NTC_5) # Add back in NTC control
p5_foo <- merge_phyloseq(otu_table(p5), tax_table(p5))
p5_foo <- merge_phyloseq(p5_foo, sample_data(p5))

# Merge
p45 <- merge_phyloseq(p4_foo,p5_foo)

# Export taxa_names to build new tree
#taxa <- unique(c(taxa_names(p),taxa_names(p45)))
#write.csv(taxa, file = "soil.health.grab.rep.seqs.for.tree.csv", row.names = F)

# Fix sample names
sample_names(p45) <- unlist(as(sample_data(p45)[,"SampleID"], "vector"))

# Check for duplicates
intersect(sample_names(p), sample_names(p45))
sample_names(p45)[which(sample_names(p45) == "346")] <- "346_re"

# Fix columns
remove_me <- setdiff(colnames(sample_data(p45)),colnames(sample_data(p)))
setdiff(colnames(sample_data(p)), colnames(sample_data(p45)))
sample_data(p45)[which(colnames(sample_data(p45)) %in% remove_me)] <- NULL

# Merge p45 and p
p_final <- merge_phyloseq(p_foo, p45)

# Add in tree
p_tree <- readRDS(file = "soil.health.rep.seqs.rooted.tree.rds")
p_final <- merge_phyloseq(p_final, p_tree)

# Keep information about the total # of sequences and OTUs prior to filtering for later
total_seqs <- sum(sample_sums(p))
total_otus <- length(taxa_names(p))

# Denote Contaminants in No Template Controls | The function works, but when called here, phyloseq throws an error (super frustrating)
blanks <- subset_samples(p_final, sample_names(p_final) == "NTC.control" | sample_names(p_final) == "NTC_1" | sample_names(p_final) == "NTC_2")
blanks <- subset_taxa(blanks, taxa_sums(blanks) > 0)
blanks <- subset_taxa(p_final, taxa_names(p_final) %in% taxa_names(blanks))

for (rank in colnames(tax_table(blanks))){
  tax_table(blanks)[,rank] <- paste(tax_table(blanks)[,rank],"**possible contaminant**",sep=" ")
}

all_others <- prune_taxa(setdiff(taxa_names(p_final), taxa_names(blanks)), p_final)
all_others <- tax_table(all_others)
full <- rbind(all_others, tax_table(blanks))
full <- full[order(rownames(full)),]

tax_table(p_final) <- tax_table(full)
p_final <- subset_samples(p_final, sample_names(p_final) != "NTC.control" & sample_names(p_final) != "NTC_1" & sample_names(p_final) != "NTC_2")

## Sequences lost to NTC
#print(paste("You've lost ",round((1-(sum(sample_sums(p_filtered))/total_seqs))*100,2),"% of your total sequences after removing contaminants found in NTC.", sep=""))
# "You've lost 1.07% of your total sequences after removing contaminants found in NTC."
#print(paste("You've lost ",round((1-(length(taxa_names(p_filtered))/total_otus))*100,1),"% of your total OTUs (n=",length(taxa_names(p_blanks)),") sequences after removing contaminants found in NTC.", sep=""))
# "You've lost 0.1% of your total OTUs (n=118) sequences after removing contaminants found in NTC."

######################################################################################
# Add in missing lat. and long. data (Kirsten provided this after sequence processing)
x.geo <- read.csv(file = "full.lat.long.csv", header = T, stringsAsFactors = F)
x.geo <- merge(data.frame(sampleID = sample_names(p_final)), x.geo, by = "sampleID") # Subset to those present in p
x.geo <- x.geo[match(sample_names(p_final), x.geo$sampleID),]

# fix random latitude with a space
x.geo[which(x.geo$sampleID == 364), "Latitude"] <- gsub(" ","", x.geo[which(x.geo$sampleID == 364), "Latitude"])

# Fix Longitude (missing a negative for the western hemisphere)
x.geo$Latitude <- as.numeric(x.geo$Latitude)
x.geo$Longitude <- as.numeric(x.geo$Longitude)
x.geo$Longitude[which(x.geo$Longitude > 0)] <- x.geo$Longitude[which(x.geo$Longitude > 0)]*-1

# Make x.geo lat. long for 
sample_data(p_final)[,c("Latitude","Longitude")] <- x.geo[,c("Latitude","Longitude")]

# Note: Samples 757 - 763 and 786 - 835 were missing lat and long info
# Lat/Long were estimated from the ZIP CODE provided

# Sanity check (make sure all longitude is in the western hemisphere)
# any(x.geo$Longitude > 0)

#####################################################
# Add in missing soil health data for samples 982-998

foo <- subset_samples(p_final, sample_names(p_final) %in% c(982:998))
#write.csv(as.data.frame(as(sample_data(foo), "matrix"), stringsAsFactors = F), file = "add.in.extras.final.csv")
fixed <- read.csv(file = "add.in.extras.final.csv", stringsAsFactors = F, header = T)
rownames(fixed) <- fixed$X
fixed$X <- NULL
sample_data(foo) <- sample_data(fixed)
p_final <- merge_phyloseq(subset_samples(p_final, !(sample_names(p_final) %in% c(982:998))),foo)


#######################################################################
# Fix missing pH ratings (provided by Joseph Amsili or Bob Shindelbeck)
pH <- read.csv(file = "updated.soil.health.pH.ratings.csv", header = T, stringsAsFactors = F)
foo <- subset(pH, sampleID %in% sample_names(p_final))
foo <- foo[match(sample_names(p_final), foo$sampleID), ]
sample_data(p_final)[,"pH"] <- foo$pH
sample_data(p_final)[,"pH_rating"] <- foo$pH.score


######################################
# Calculate average soil health rating
soil.health.ratings <- as.data.frame(as(sample_data(p_final)[,grep("rating",colnames(sample_data(p_final)))], "matrix"))
avg.ratings <- apply(soil.health.ratings, 1, function(x) mean(x, na.rm=T))
#hist(avg.ratings)
#shapiro.test(avg.ratings)  #average rating is normally distributed
sample_data(p_final)[,"avg.rating"] <- avg.ratings
sample_data(p_final)[,"health.category"]<- cut(sample_data(p_final)$avg.rating, c(0,20,40,60,80,100))

# Include ratings as categorical variables
for (rating in c("water_capacity_rating","surface_hardness_rating","subsurface_hardness_rating","aggregate_stability_rating","organic_matter_rating","ace_soil_protein_index_rating","respiration_rating","active_carbon_rating","pH_rating","P_rating","K_rating","minor_elements_rating")){
  foo <- cut(as.numeric(as(sample_data(p_final)[,rating], "matrix")), c(0,20,40,60,80,100)) # cut can't handle zeros
  foo[is.na(foo)] <- "(0,20]"
  sample_data(p_final)[,gsub("_rating",".category",rating)] <- foo
  
}

##############################
# Add in geographical grouping
p_final <- geo_group(p_final, proximity)

# Use Field ID for more specific geogroup assignments for large sample 
foo <- data.frame(as(sample_data(p_final), "matrix"), stringsAsFactors = F)
foo$latlon <- paste(foo$Latitude, foo$Longitude, sep="_")
unique(subset(foo, latlon %in% names(table(foo$latlon)[which(table(foo$latlon)>5)]))$geo_group)

## Use FieldIDs for: 10,25,59,92 and 112

# geo_groups 10, 25 and 59
for (group in c("10","25","59")){
  geo <- subset(foo, geo_group == group)[,c("fieldID","avg.rating")]
  geo$fieldID <- gsub("-1$|-2$|-3$|-4$","",geo$fieldID)
  geo$fieldID <- gsub("Species","",geo$fieldID)
  geo$geo_group <- refactor(geo, "fieldID")+max(as.numeric(as(unique(sample_data(p_final)[,"geo_group"]),"matrix")))
  sample_data(p_final)[which(sample_names(p_final) %in% rownames(subset(foo, geo_group == group))), "geo_group"] <- geo$geo_group
  
}

# geo_group 92
fai <- na.omit(subset(foo, geo_group == "92")[,c("fieldID","avg.rating")])
fai$fieldID <- gsub("PMC2017AK2170","",fai$fieldID)
fai$fieldID <- gsub("T\\d|T\\d\\d","",fai$fieldID)
fai$geo_group <- refactor(fai, "fieldID")+max(as.numeric(as(unique(sample_data(p_final)[,"geo_group"]),"matrix")))
sample_data(p_final)[which(sample_names(p_final) %in% rownames(fai)), "geo_group"] <- fai$geo_group

# geo_group 112
fai <- na.omit(subset(foo, geo_group == "112")[,c("fieldID","avg.rating")])
fai$fieldID <- gsub("\\d|\\d\\d|-\\d","",fai$fieldID)
fai$geo_group <- refactor(fai, "fieldID")+max(as.numeric(as(unique(sample_data(p_final)[,"geo_group"]),"matrix")))
sample_data(p_final)[which(sample_names(p_final) %in% rownames(fai)), "geo_group"] <- fai$geo_group

######################################
## Select one for each repeated sample

## Compare repeats
## Select which repeats to keep (all of the repeats with higher sequencing depth)
first <- c("542","63","346","789","835","873","930","981")
second <- sample_names(p_final)[grep("re",sample_names(p_final))]
repeats <- subset_samples(p_final, sample_names(p_final) %in% c(first, second))

# Save repeats for BLOCK02
saveRDS(repeats, file = "p_soil.health.repeats.filter.norm.rds")

# remove unneeded repeats
keepers <- c("346_re", "542_repeat","63_repeat","789_re","835","873_reextract","930_re","981")
remove_me <- setdiff(c(first, second), keepers)

p_final <- prune_samples(setdiff(sample_names(p_final),remove_me), p_final)
p_final <- subset_taxa(p_final, taxa_sums(p_final) > 0)

# Ensure the proper transmission of sample data (some repeats lack sample data)
sample_names(p_final)[grep("346_re", sample_names(p_final))] <- "346"
sample_names(p_final)[grep("542_repeat", sample_names(p_final))] <- "542"
sample_names(p_final)[grep("63_repeat", sample_names(p_final))] <- "63"
sample_names(p_final)[grep("789_re", sample_names(p_final))] <- "789"
sample_names(p_final)[grep("873_reextract", sample_names(p_final))] <- "873"
sample_names(p_final)[grep("930_re", sample_names(p_final))] <- "930"

samples <- as.data.frame(as(sample_data(p_final), "matrix"), stringsAsFactors = F)
samples[grep("346$", rownames(samples)),] <- as.data.frame(as(sample_data(repeats)[grep("346$", sample_names(repeats))], "matrix"), stringsAsFactors = F)
samples[grep("542$", rownames(samples)),] <- as.data.frame(as(sample_data(repeats)[grep("542$", sample_names(repeats))], "matrix"), stringsAsFactors = F)
samples[grep("^63$", rownames(samples)),] <- as.data.frame(as(sample_data(repeats)[grep("^63$", sample_names(repeats))], "matrix"), stringsAsFactors = F)
samples[grep("789$", rownames(samples)),] <- as.data.frame(as(sample_data(repeats)[grep("789$", sample_names(repeats))], "matrix"), stringsAsFactors = F)
samples[grep("873$", rownames(samples)),] <- as.data.frame(as(sample_data(repeats)[grep("873$", sample_names(repeats))], "matrix"), stringsAsFactors = F)
samples[grep("930$", rownames(samples)),] <- as.data.frame(as(sample_data(repeats)[grep("930$", sample_names(repeats))], "matrix"), stringsAsFactors = F)

# we have samples from two time_points for geo group "5" (and one was marked tillage '1')
# it is a rotation of soy, wheat and grain corn (COG)
# attribute tillage class "1" to all
samples[which(samples$geo_group == "5"),"tillage"] <- "1"

# same goes for Rocco's. They have one sample listed as no-till and correctly assigned "1", but their "CT" (Assumed conventional till) is not given a tillage group
samples[which(samples$geo_group == "41" & samples[,"fieldID"] == "Rocco Lewis CT Corn Silage"),"tillage"] <- "2"

########################################################################
## Create ID for geo_group divided by management practice (i.e. tillage)

samples$manage_group <- paste(samples$geo_group, samples$tillage, sep="_")

# Sanity check (any geo_gropus with tillage defined in some, but not others)
#subset(manage_group, geo_group %in% unique(manage_group[grep("NA",manage_group$m.group),"geo_group"]))

########################
# Amend main data object
sample_data(p_final) <- sample_data(samples)

# Convert characters to numeric in sample data
for (col in c("Latitude","Longitude","P_rating","K","K_rating","Mg","Fe","Mn","Zn","minor_elements_rating","avg.rating","mean.copies.per.g","DNA","respiration","respiration_rating","active_carbon","active_carbon_rating","pH","pH_rating","P","organic_matter","organic_matter_rating","ace_soil_protein_index","ace_soil_protein_index_rating","surface_hardness_rating","subsurface_hardness","subsurface_hardness_rating","aggregate_stability","aggregate_stability_rating","soil_texture_sand","soil_texture_silt","soil_texture_clay","water_capacity","water_capacity_rating","surface_hardness")){
  sample_data(p_final)[,col] <- as.numeric(as(sample_data(p_final)[,col], "matrix"))
}

# Save Output
saveRDS(p_final, file = "p_SSU.soil.health.silva.june.2019.rds")


#####################
## Sparsity Filtering

# Hard filter low samples with low read-depth
p_filtered <- prune_samples(names(sample_sums(p_final)[which(sample_sums(p_final) > 2000)]), p_final)  

# Remove OTUs with zero counts
p_filtered <- subset_taxa(p_filtered, taxa_sums(p_filtered) > 0)

# Save minimum filtered dataset and Aurora validation dataset separately
saveRDS(subset_samples(p_filtered, !(sample_names(p_filtered) %in% 961:981)), file = "p_SSU.soil.health.minimal.rds")
saveRDS(subset_samples(p_filtered, sample_names(p_filtered) %in% 961:981), file = "p_SSU.soil.health.aurora.minimal.rds")

# Make minimum filtered normalized dataset
foo <- p_filtered
sample_totals <- sample_sums(foo)
otu_table(foo) <- otu_table(normalize_to_cpt(foo), taxa_are_rows = TRUE)
saveRDS(subset_samples(foo, !(sample_names(foo) %in% 961:981)), file = "p_SSU.soil.health.minimal.norm.rds")
saveRDS(subset_samples(foo, sample_names(foo) %in% 961:981), file = "p_SSU.soil.health.aurora.minimal.norm.rds")

## Figure out what depth to rarefy to.
#sample_sums(p_filtered)[order(sample_sums(p_filtered))]
#sample_data(subset_samples(p_filtered, sample_names(p_filtered) %in% c("135","916","527","110")))
#table(sample_data(p_filtered)$geo_group)

# Rarefy to the third smallest library b/c the lowest two have more than 1 member of the geogroup
minimum <- rarefy_even_depth(p_filtered, rngseed = 42, replace = FALSE, trimOTUs = TRUE, verbose = TRUE)

# Save output
saveRDS(subset_samples(minimum, !(sample_names(minimum) %in% 961:981)), file = "p_SSU.soil.health.minimal.rare.rds")
saveRDS(subset_samples(minimum, sample_names(minimum) %in% 961:981), file = "p_SSU.soil.health.aurora.minimal.rare.rds")


# Hard Filter all singleton OTUs (there are only doubletons in DADA2)
p_filtered <- subset_taxa(p_filtered, taxa_sums(p_filtered) > 2.5)  

## Soft Filtering - Keep ASVs present in > three samples
##
present_absent<-otu_table(p_filtered) 
present_absent[present_absent > 0] <- 1
p_filtered <- prune_taxa(rownames(present_absent)[which(rowSums(present_absent) >= 3)], p_filtered)

# Make separate Date for Chloroplast, Mitochondria and unclassifieds
euk <- taxa_names(subset_taxa(p_filtered, Class == "Chloroplast" | Family == "Mitochondria" | Domain == "Eukaryota" | Domain == "putative Unassigned" | Domain == "Unassigned" | Domain == "putative Eukaryota"))
p_nonbact <- prune_taxa(euk, p_filtered)

# save Bacteria as main object
p_filtered <- prune_taxa(setdiff(taxa_names(p_filtered), taxa_names(p_nonbact)), p_filtered)

# Save filtered
saveRDS(subset_samples(p_filtered, !(sample_names(p_filtered) %in% 961:981)), file = "p_SSU.soil.health.filtered.rds")  
saveRDS(subset_samples(p_filtered, sample_names(p_filtered) %in% 961:981), file = "p_SSU.soil.health.aurora.filtered.rds")
saveRDS(p_nonbact, file = "p_non.bacteria.soil.health.filtered.rds")

set.seed(42)
p_filtered_rare <- rarefy_even_depth(p_filtered, sample.size = min(sample_sums(p_filtered)), rngseed = 42, replace = FALSE, trimOTUs = TRUE, verbose = TRUE)
saveRDS(subset_samples(p_filtered_rare, !(sample_names(p_filtered_rare) %in% 961:981)), file = "p_SSU.soil.health.filter.rare.rds")  
saveRDS(subset_samples(p_filtered_rare, sample_names(p_filtered_rare) %in% 961:981), file = "p_SSU.soil.health.aurora.filter.rare.rds")

# Normalize Abundances - Cumulative Sum Scaling
saveRDS(calc.css(subset_samples(p_filtered, !(sample_names(p_filtered) %in% 961:981))), file = "p_SSU.soil.health.filter.css.rds")  
saveRDS(calc.css(subset_samples(p_filtered, sample_names(p_filtered) %in% 961:981)), file = "p_SSU.soil.health.aurora.filter.css.rds")  

# Normalize Abundances - Scaled to Counts Per Thousand Reads
sample_totals <- sample_sums(p_filtered)
otu_table(p_filtered) <- otu_table(normalize_to_cpt(p_filtered), taxa_are_rows = TRUE)

# Normalize non-Bacteria
x.euk <- as.data.frame(as(otu_table(p_nonbact), "matrix"), stringsAsFactors = F)  

for (n in 1:nrow(x.euk)){
  x.euk[n,] <- x.euk[n,]/(sample_totals/1000)
}
otu_table(p_nonbact) <- otu_table(x.euk, taxa_are_rows = TRUE)

# Save final objects
saveRDS(subset_samples(p_filtered, !(sample_names(p_filtered) %in% 961:981)), file = "p_SSU.soil.health.filter.norm.rds")  
saveRDS(subset_samples(p_filtered, sample_names(p_filtered) %in% 961:981), file = "p_SSU.soil.health.aurora.filter.norm.rds") 


###
### Save 'psmelt'  (Done on Server) 
source("get_dataset.R")

for (dataset in c("minimal","minimal.rare","minimal.norm","filtered","filter.rare","filter.css","filter.norm")){
  for (set in c("full","aurora")){
    p <- get_dataset(dataset, set)
    x <- psmelt(p)
    
    if (set == "full"){
      if (dataset == "minimal"){
        saveRDS(x, file = "x.SSU.soil.health.minimal.melt.rds")    
        
      } else if (dataset == "minimal.rare") {
        saveRDS(x, file = "x.SSU.soil.health.minimal.rare.melt.rds")    
        
      } else if (dataset == "minimal.norm") {
        saveRDS(x, file = "x.SSU.soil.health.minimal.norm.melt.rds")    
        
      } else if (dataset == "filtered") {
        saveRDS(x, file = "x.SSU.soil.health.filter.melt.rds")    
        
      } else if (dataset == "filter.rare") {
        saveRDS(x, file = "x.SSU.soil.health.filter.rare.melt.rds")    
        
      } else if (dataset == "filter.css") {
        saveRDS(x, file = "x.SSU.soil.health.filter.css.melt.rds")    
        
      } else if (dataset == "filter.norm"){
        saveRDS(x, file = "x.SSU.soil.health.filter.norm.melt.rds")    
      }
    } else if (set == "aurora"){
      if (dataset == "minimal"){
        saveRDS(x, file = "x.SSU.soil.health.aurora.minimal.melt.rds")    
        
      } else if (dataset == "minimal.rare") {
        saveRDS(x, file = "x.SSU.soil.health.aurora.minimal.rare.melt.rds")    
        
      } else if (dataset == "minimal.norm") {
        saveRDS(x, file = "x.SSU.soil.health.aurora.minimal.norm.melt.rds")    
        
      } else if (dataset == "filtered") {
        saveRDS(x, file = "x.SSU.soil.health.aurora.filter.melt.rds")    
        
      } else if (dataset == "filter.rare") {
        saveRDS(x, file = "x.SSU.soil.health.aurora.filter.rare.melt.rds")    
        
      } else if (dataset == "filter.css") {
        saveRDS(x, file = "x.SSU.soil.health.aurora.filter.css.melt.rds")    
        
      } else if (dataset == "filter.norm"){
        saveRDS(x, file = "x.SSU.soil.health.aurora.filter.norm.melt.rds")    
      }
    }
    
    rm(p)
    rm(x)
    
  }
}

###
### Aggregate counts by taxonomy
for (dataset in c("minimal","minimal.rare","minimal.norm","filter","filter.rare","filter.css","filter.norm")){
  for (set in c("full","aurora")){
    if (set == "full"){
      x <- readRDS(file = paste("x.SSU.soil.health.",dataset,".melt.rds",sep=""))
    } else if (set == "aurora"){
      x <- readRDS(file = paste("x.SSU.soil.health.aurora.",dataset,".melt.rds",sep=""))
    }
  }

  # Tally all counts for each rank
  for (rank in c("Phylum","Class","Order","Family","Genus")){
    formula <- as.formula(paste("~",paste(c(rank,"Sample"), collapse = "+"), sep=""))
    if (set == "full"){
      saveRDS(ddply(x, formula, summarise, Total.Abundance = sum(Abundance)), file = paste("x.SSU.soil.health",dataset,"melt",rank,"rds", sep="."))
    } else if (set == "aurora"){
      saveRDS(ddply(x, formula, summarise, Total.Abundance = sum(Abundance)), file = paste("x.SSU.soil.health.aurora",dataset,"melt",rank,"rds", sep="."))
    }
  }
  rm(x)
}