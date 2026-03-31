#Import Dataset Phyloseq 
get_dataset <- function(stage, set) {
  if (set == "full"){
    if (stage == "original"){
      p <- readRDS(file = "../microbiome.data/p_SSU.soil.health.final.rds")
      
    } else if (stage == "minimal"){
      p <- readRDS(file = "../microbiome.data/p_SSU.soil.health.minimal.rds")    
      
    } else if (stage == "minimal.rare"){
      p <- readRDS(file = "../microbiome.data/p_SSU.soil.health.minimal.rare.rds")    
      
    } else if (stage == "filtered"){
      p <- readRDS(file = "../microbiome.data/p_SSU.soil.health.filtered.rds") 
      
    } else if (stage == "filter.rare"){
      p <- readRDS(file = "../microbiome.data/p_SSU.soil.health.filter.rare.rds") 
      
    } else if (stage == "filter.css"){
      p <- readRDS(file = "../microbiome.data/p_SSU.soil.health.filter.css.rds") 
      
    } else if (stage == "filter.norm"){
      p <- readRDS(file = "../microbiome.data/p_SSU.soil.health.filter.norm.rds")    
    }
  } else if (set == "aurora"){
    if (stage == "minimal"){
      p <- readRDS(file = "../microbiome.data/p_SSU.soil.health.aurora.minimal.rds")    
      
    } else if (stage == "minimal.rare"){
      p <- readRDS(file = "../microbiome.data/p_SSU.soil.health.aurora.minimal.rare.rds")    
      
    } else if (stage == "filtered"){
      p <- readRDS(file = "../microbiome.data/p_SSU.soil.health.aurora.filtered.rds") 
      
    } else if (stage == "filter.rare"){
      p <- readRDS(file = "../microbiome.data/p_SSU.soil.health.aurora.filter.rare.rds") 
      
    } else if (stage == "filter.css"){
      p <- readRDS(file = "../microbiome.data/p_SSU.soil.health.aurora.filter.css.rds") 
      
    } else if (stage == "filter.norm"){
      p <- readRDS(file = "../microbiome.data/p_SSU.soil.health.aurora.filter.norm.rds")    
    }
  }
  
  return(p)
}