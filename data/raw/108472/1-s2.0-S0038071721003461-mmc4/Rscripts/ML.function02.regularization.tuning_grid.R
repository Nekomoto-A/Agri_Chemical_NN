## Taken from model_selection.R from Topcuoglu et al., 2020

# load libraries
deps = c("dplyr", "tictoc", "caret" ,"rpart", "randomForest", "kernlab","LiblineaR", "pROC", "tidyverse");
for (dep in deps){
  library(dep, verbose=FALSE, character.only=TRUE)
}

tuning_grid <- function(trainTransformed, model, factor){
  folds <- 5
  cvIndex <- createMultiFolds(factor(trainTransformed[,factor]), folds, times=100)
  
  if (length(unique(trainTransformed[,factor])) == 2){
    if (model == "L2LinearSVM"){
      cv <- trainControl(method="repeatedcv",
                         number=folds,
                         index = cvIndex,
                         returnResamp="final",
                         summaryFunction=twoClassSummary,
                         indexFinal=NULL,
                         savePredictions = TRUE)
    } else {
      cv <- trainControl(method="repeatedcv",
                         number=folds,
                         index = cvIndex,
                         returnResamp="final",
                         classProbs=TRUE,
                         summaryFunction=twoClassSummary,
                         indexFinal=NULL,
                         savePredictions = TRUE)
    }
    
    met = "ROC"
    
  } else {
    
    if (model == "L2LinearSVM"){
      cv <- trainControl(method="repeatedcv",
                         number=folds,
                         index = cvIndex,
                         returnResamp="final",
                         summaryFunction=defaultSummary,
                         indexFinal=NULL,
                         savePredictions = TRUE)
      
    } else {
      cv <- trainControl(method="repeatedcv",
                         number=folds,
                         index = cvIndex,
                         returnResamp="final",
                         classProbs=TRUE,
                         summaryFunction=defaultSummary,
                         indexFinal=NULL,
                         savePredictions = TRUE)
      
    }
    
    met = "Accuracy"
  }
  
  # https://topepo.github.io/caret/available-models.html
  
  if (ML_approach == "regression"){
    met <- "Rsquared"
  }
  
  if(model=="L2LogisticRegression") {
    grid <-  expand.grid(cost = c(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 5),
                         loss =  "L2_primal", epsilon = 0.01) 
    method <- "regLogistic"

  } else if (model=="RandomForest"){
    grid <-  expand.grid(mtry = round(c(ncol(data)*0.1, ncol(data)*0.25, ncol(data)*0.5, ncol(data)*0.75, ncol(data))))
    method = "rf"

  } else if (model=="L2LinearSVM"){
    grid <- expand.grid(cost = c(0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 5),
                        Loss = "L2")
    method <- "svmLinear3" # This function was modified in caret by Topcuoglu et al., 2020
  }

  params <- list(grid, method, cv, met)
  return(params)
}