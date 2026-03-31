library(phyloseq)

# Note: a 'feature_set' can be either a sampleID or OTU(s).

get_geogroups <- function(feature_set, p) { 

  # Get Total
  total_geogroups <- nrow(unique(sample_data(p)[,"geo_group"]))
  
  # Determine if feature is OTU
  feature_data <- try(prune_taxa(feature_set, p), silent = TRUE)

  # Catch missing OTU
  if (class(feature_data) == "try-error") {
    
    # Assume feature is sampleID
    feature_data <- prune_samples(feature_set, p)  #Note: there is a bug if you try using 'subset_samples' here.
    missive = "sample"
    
  } else {
    missive = "OTU"
  }
  
  # Only keep sample containing taxa
  feature_data <- prune_samples(names(sample_sums(feature_data) > 0), feature_data) #Note: there is a bug if you try using 'subset_samples' here.
  
  if (missive == "OTU"){
    genera <- paste(unique(tax_table(feature_data)[,"Genus"]),collapse=",")  
  }
  
  geo_groups <- unique(sample_data(feature_data)[,"geo_group"])

  if (missive == "OTU"){
    print(paste("Your OTUs are found in a total of", nrow(geo_groups), "/", total_geogroups, "group(s) and belong to the genera:", genera, sep=" "))
  } else {
    print(paste("Your samples span a total of ", nrow(geo_groups), " / ", total_geogroups, " geo_groups.", sep=""))
  }
  
  return(geo_groups)
}