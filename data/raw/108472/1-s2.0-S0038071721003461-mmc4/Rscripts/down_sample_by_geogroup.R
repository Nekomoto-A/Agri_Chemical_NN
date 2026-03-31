library(phyloseq)

down_sample_by_geogroup <- function(phyloseq, max_samples){
  
  # Downsample according to cap
  cap <- table(sample_data(phyloseq)[,"manage_group"])[which(table(sample_data(phyloseq)[,"manage_group"]) > max_samples)]  
  p_save <- subset_samples(phyloseq, !(manage_group %in% names(cap)))
  
  if (length(cap) > 0){
    for (site in names(cap)){
      p_subset <- subset_samples(p, manage_group == site)
      p_subset <- prune_samples(sample(sample_names(p_subset), max_samples, replace=F), p_subset)
      p_save <- merge_phyloseq(p_save, p_subset)
    }
  }
  
  return(p_save)
}
