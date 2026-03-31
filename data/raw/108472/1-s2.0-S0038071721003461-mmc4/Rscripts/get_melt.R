library(reshape2)

get_melt <- function(dataset, rank, set){
  if (set == "full"){
    x <- readRDS(file = paste("../microbiome.data/melt.soil.health",dataset,rank,"rds", sep="."))
  } else if (set == "aurora"){
    x <- readRDS(file = paste("../microbiome.data/melt.soil.health.aurora",dataset,rank,"rds", sep="."))
  }
  x <- reshape(x, direction = "wide", idvar = "Sample", timevar = rank)
  colnames(x) <- gsub("Total.Abundance.","",colnames(x))
  colnames(x)[1] <- "sample"

  return(x)
}