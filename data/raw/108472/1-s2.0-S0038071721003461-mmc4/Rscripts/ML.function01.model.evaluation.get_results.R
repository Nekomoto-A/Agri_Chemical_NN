library(tidyr)

## Taken from generateAUCs.R from Topcuoglu et al., 2020
get_results <- function(data, model, seed, factor, run.name){
  
  # Save results of the modeling pipeline as a list
  results <- pipeline(data, model, seed, factor) # ML.block01

  # Save model as part of full results
  if (feature_type == "Taxonomy"){
    save_name <- paste("data/results/all.results_", run.name, "_", model,"_", seed, "_", dataset, "_", factor, "_", rank, ".rds", sep="")
  } else {
    save_name <- paste("data/results/all.results_", run.name, "_", model,"_", seed, "_", dataset, "_", factor, ".rds", sep="")
  }
  saveRDS(results, file = save_name)
  
  ## Create a matrix with cv_metrics and test_metrics from 1 data split
  metrics <- matrix(c(results[[1]], results[[2]]), ncol=2)
  
  if (full_analysis == "T"){
    evaluation_metric <- results[[7]]
  } else {
    evaluation_metric <- results[[5]]
  }
  
  # Convert to dataframe and add a column noting the model name
  metrics_dataframe <- data.frame(metrics) %>%
    mutate(model=model) %>%
    add_column(metric=evaluation_metric)
  colnames(metrics_dataframe)[1:2] <- c("cv_metric","test_metric") 
  
  if (feature_type == "Taxonomy"){
    save_name <- paste0("data/results/best_hp_results_", run.name, "_", model,"_", seed, "_", dataset, "_", factor, "_", rank, ".csv")
  } else {
    save_name <- paste0("data/results/best_hp_results_", run.name, "_", model,"_", seed, "_", dataset, "_", factor, ".csv")
  }
  write_csv(metrics_dataframe, path=save_name, col_names = TRUE)
  
  ## Save results for all hyper-parameters for 1 datasplit and corresponding metrics
  all_results <- results[3]
  
  # Convert to dataframe and add a column noting the model name
  dataframe <- data.frame(all_results) %>%
    mutate(model=model)  
  
  if (feature_type == "Taxonomy"){
    save_name <- paste0("data/results/all_hp_results_", run.name, "_", model,"_", seed,"_", dataset, "_", factor,"_", rank, ".csv")
  } else {
    save_name <- paste0("data/results/all_hp_results_", run.name, "_", model,"_", seed,"_", dataset, "_", factor, ".csv")
  }
  write_csv(dataframe, path=save_name, col_names = TRUE)
}