
# Cumulative Sum Scaling (CSS) Normalization Code
# See publication 10.1038/nmeth.2658 for how it works
# Code provided by Allison Thompson (allison.thompson@pnnl.gov)

# Define CSS normalization function
# This is set to 75th percentile but can be changed

CSS_Norm <- function(e_data, edata_id, q=0.75, qg="median"){
  e_data_norm <- e_data[,-which(colnames(e_data)==edata_id)]
  e_data_norm[e_data_norm==0] <- NA
  
  # calculate the q quantile of data, per sample and globally #
  col.q = apply(e_data_norm, 2, function(x) sum(x[x<=quantile(x, probs = q, na.rm=TRUE)], na.rm=TRUE))
  
  if(qg == 1000){
    g.q = 1000
  }else if(qg == "median"){
    g.q = median(col.q, na.rm=TRUE)
  }else{
    stop("Invalid value for qg")
  }
  #g.q = 1000
  #g.q = median(col.q, na.rm=TRUE)
  
  # normalize omics_data data by q quantile and transform back to count data #
  for(i in 1:ncol(e_data_norm)){
    e_data_norm[,i] = (e_data_norm[,i] / col.q[i]) * g.q
  }
  
  e_data <- data.frame(e_data[,which(colnames(e_data)==edata_id)],e_data_norm)
  colnames(e_data)[1] <- edata_id
  e_data[is.na(e_data)] <- 0
  return(e_data)
}