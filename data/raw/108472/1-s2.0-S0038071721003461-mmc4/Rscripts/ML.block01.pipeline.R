pipeline <- function(data, model, seed, factor, full_analysis){
  
  # ------------------Pre-process the full data------------------------->
  # We are doing the pre-processing to the full data and then splitting 80-20
  # Scale all features between 0-1
  preProcValues <- preProcess(data, method = "range")
  dataTransformed <- predict(preProcValues, data)
  
  # ------------------80-20 Datasplit for each seed------------------------->
  if (ML_approach == "regression"){
    inTraining <- createDataPartition(as.numeric(dataTransformed[,factor]), groups = 4,  p = .80, list = FALSE)
  } else {
    inTraining <- createDataPartition(dataTransformed[,factor], p = .80, list = FALSE)
  }
  
  trainTransformed <- dataTransformed[ inTraining,]
  testTransformed  <- dataTransformed[-inTraining,]

  # -------------Define hyper-parameter and cv settings-------------------->
  grid <- tuning_grid(trainTransformed, model, factor)[[1]]   # function02
  method <- tuning_grid(trainTransformed, model, factor)[[2]] # function02
  cv <- tuning_grid(trainTransformed, model, factor)[[3]]     # function02
  met <- tuning_grid(trainTransformed, model, factor)[[4]]    # function02
  
  # Information on evaluation metrics for ML in https://machinelearningmastery.com/machine-learning-evaluation-metrics-in-r/

  # ---------------------------Train the model ---------------------------->
  # Start walltime
  tic("train")
  
  if(model=="L2LogisticRegression"){
    print(model)
    trained_model <-  train(as.formula(paste(factor,"~ .")), # label
                            data=trainTransformed, #total data
                            method = method,
                            trControl = cv,
                            metric = met,
                            tuneGrid = grid,
                            family = "binomial")
    
  } else if (model=="RandomForest"){
    print(model)
    trained_model <-  train(as.formula(paste(factor,"~ .")),
                            data=trainTransformed,
                            method = method,
                            trControl = cv,
                            metric = met,
                            tuneGrid = grid,
                            ntree=1000) # not tuning ntree
  } else {
    print(model)
    if (ML_approach == "classification"){
      trained_model <-  train(as.formula(paste(factor,"~ .")),
                              data=trainTransformed,
                              method = method,
                              trControl = cv,
                              metric = met,
                              tuneGrid = grid)
      
    } else if (ML_approach == "regression"){
      trained_model <-  train(as.formula(paste(factor,"~ .")),
                              data=trainTransformed,
                              method = method,
                              trControl = cv,
                              metric = met,
                              svr_eps = 0.1,
                              tuneGrid = grid)
    }
  }

  # Save wall-time
  seconds <- toc()
  train_time <- seconds$toc-seconds$tic
  
  if (feature_type == "Taxonomy"){
    write.csv(train_time, file=paste0("data/walltime/traintime_", run.name, "_", model, "_", seed, "_", dataset, "_", factor, "_", rank, ".csv"), row.names=F)
  } else {
    write.csv(train_time, file=paste0("data/walltime/traintime_", run.name, "_", model, "_", seed, "_", dataset, "_", factor, ".csv"), row.names=F)
  }

  if (met == "Accuracy" & ML_approach == "classification"){
    eval_measure <- "Kappa"
  } else if (ML_approach == "regression"){
    eval_measure <- "Rsquared"
  }
  
  # ------------- Output the evaluation metric (Accuracy | Rsquared) ---------------------->
  # Mean cv AUC or mean cv Accuracy  over repeats of the best cost parameter during training
  cv_metric <- getTrainPerf(trained_model)[,paste("Train",eval_measure,sep="")]

  # Save all results of hyper-parameters and their corresponding metrics over 100 internal repeats
  results_individual <- trained_model$results
  
  # Return all the metrics
  results <- list(cv_metric, test_metric, results_individual, trained_model, eval_measure)

  return(results)
}
