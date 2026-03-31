get_study <- function(stage, set) {
  if (stage == "original"){
    p <- readRDS(file = "../microbiome.data/p_SSU.soil.health.final.rds")
    
  } else if (set == "aurora"){
    p <- readRDS(file = paste("../microbiome.data/p_SSU.soil.health.aurora.", stage,".rds",sep=""))    
    
  } else {
    p <- readRDS(file = paste("../microbiome.data/p_SSU.soil.health.", stage,".rds",sep=""))    
  } 
  return(p)
}