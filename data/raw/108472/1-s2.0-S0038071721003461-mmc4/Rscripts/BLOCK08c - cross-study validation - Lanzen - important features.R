library(phyloseq)
source('get_study.R')

###
### Identify Overlap between Important Features and Features in Lanzen

# Import Lanzen microbiome data
p <-  readRDS(file = "../microbiome.data/p_lanzen.final.rds")
p <- subset_taxa(p, taxa_sums(p) > 0)

# Total OTU overlap
v <- readRDS(file = "../microbiome.data/minimal.ASV.datset.rds") # from BLOCK03
length(intersect(taxa_names(p), taxa_names(v)))
length(taxa_names(v))

# Import Feature Importance data for SH models
x <- readRDS(file = "../models/compiled.feature.selection.results.final.rds") # output from "ML.block06a.feature.importance.SH.R"
x <- subset(x, rank == "ASV")
x <- x[order(x$ratio),]

result <- data.frame(health.metric = NA, overlap = NA, stringsAsFactors = F)
for (metric in unique(x$health.metric)){
  foo <- subset(x, health.metric == metric & dataset == "minimal.norm" & model == "RandomForest" & ratio < 0.998)
  
  if (nrow(foo) > 0){
    olap <- length(taxa_names(subset_taxa(p, taxa_names(p) %in% foo$names)))
    result <- rbind(result, data.frame(health.metric = metric, overlap = olap, stringsAsFactors = F))
  } else {
    result <- rbind(result, data.frame(health.metric = metric, overlap = NA, stringsAsFactors = F))
  }
}

write.csv(result, file = "figures/overlapping.features.lanzen.csv", quote = F, row.names = F)



###
### Identify Overlap between *Important* Features in SH models and *Important* Features in Lanzen

# Import and filter Lanzen
x.lanz <- readRDS(file = "../models/lanzen.feature.importance.final.rds")
x.lanz <- subset(x.lanz, rank == "ASV" & dataset == "minimal.norm")
x.lanz <- x.lanz[order(x.lanz$ratio),]
x.lanz <- subset(x.lanz, ratio < 0.998 & basemetric > 0.1)
x.lanz <- unique(x.lanz[,c("model","health.metric","names")])

# Import and filter SH
x.SH <- readRDS(file = "../models/compiled.feature.selection.results.final.rds") # output from "ML.block06.feature.importance.SH.R"
x.SH <- subset(x.SH, rank == "ASV" & dataset == "minimal.norm")
x.SH <- x.SH[-grep("category",x.SH$health.metric),]
x.SH <- x.SH[order(x.SH$ratio),]
x.SH <- subset(x.SH, ratio < 0.998)
x.SH <- unique(x.SH[,c("model","health.metric","names")])

results <- data.frame(model = NA, health.metric = NA, overlap = NA, total.lanz = NA, total.SH = NA, stringsAsFactors = F)
for (mod in c("L2LinearSVM","RandomForest")){
  for (metric in unique(x.lanz$health.metric)){
    foo <- subset(x.lanz, health.metric == metric & model == mod)
    foo.SH <- subset(x.SH, health.metric == metric & model == mod)
    
    if (nrow(foo) > 0){
      olap <- length(which(unique(foo$names) %in% foo.SH$names))
      results <- rbind(results, data.frame(model = mod, health.metric = metric, overlap = olap, total.lanz = length(unique(foo$names)), total.SH = length(unique(foo.SH$names)), stringsAsFactors = F))
    } else {
      results <- rbind(results, data.frame(model = mod, health.metric = metric, overlap = NA,  total.lanz = NA, total.SH = NA, stringsAsFactors = F))
    }
  }
}

results <- results[complete.cases(results),]
results$percent.lanz <- round((results$overlap/results$total.lanz)*100,1)
results$percent.SH <- round((results$overlap/results$total.SH)*100,1)
results$percent.metric <- round((results$overlap/(results$total.SH+results$total.lanz-results$overlap))*100,1)
results <- results[rev(order(results$model, results$percent.metric)),]
results
write.csv(results, file = "figures/overlap.important.features.lanzen.csv", quote = F, row.names = F)