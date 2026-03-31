library(plyr)
library(reshape2)
library(combinat)
library(Hmisc)

lmp <- function (modelobject) {
  if (class(modelobject) != "lm") stop("Not an object of class 'lm' ")
  f <- summary(modelobject)$fstatistic
  p <- pf(f[1],f[2],f[3],lower.tail=F)
  attributes(p) <- NULL
  return(p)
}

# Import all prediction / observed data points
x <- readRDS(file = "../models/SH.predictions.eval.for.plotting.rds")
x <- subset(x, dataset == "minimal.norm" & rank == "ASV" & model == "RandomForest")

# Run Correlations between relative abundance of ASV and taxa with soil health ratings.
#physical.cat <- c("water_capacity.category", "surface_hardness.category", "subsurface_hardness.category", "aggregate_stability.category")  # Note: the hardness measures are present for only 1/3 or samples
#chemical.cat <- c("pH.category","P.category","K.category","minor_elements.category")
#biological.cat <- c("DNA","organic_matter.category","ace_soil_protein_index.category","respiration.category","active_carbon.category")
#other.cat <- c("tillage","soil_texture_class")

# sanity check
#setdiff(c(physical.cat,chemical.cat,biological.cat), unique(x$factor))
# note: Potassium category missing b/c of extreme unevenness

physical.rat <- c("water_capacity_rating", "surface_hardness_rating", "subsurface_hardness_rating", "aggregate_stability_rating")  # Note: the hardness measures are present for only 1/3 or samples
chemical.rat <- c("pH_rating","P_rating","K_rating")
biological.rat <- c("DNA","organic_matter_rating","ace_soil_protein_index_rating","respiration_rating","active_carbon_rating")
#other.rat <- c("soil_texture_clay","soil_texture_sand","soil_texture_silt")

# sanity check
#setdiff(c(physical.rat,chemical.rat,biological.rat), unique(x$factor))
all.rat <- c(physical.rat,chemical.rat,biological.rat,"avg.rating")
all.comb <- as.data.frame(t(combn(all.rat,2)))

for (n in 1:nrow(all.comb)){
  met1  <- all.comb[n,1]
  met2  <- all.comb[n,2]
  
  foo1 <- merge(subset(x, factor == met1)[,c("factor","seed","sampleID","obs.data")], subset(x, factor == met2)[,c("factor","seed","sampleID","predicted.data")], by = c("sampleID","seed"))
  foo2 <- merge(subset(x, factor == met2)[,c("factor","seed","sampleID","obs.data")], subset(x, factor == met1)[,c("factor","seed","sampleID","predicted.data")], by = c("sampleID","seed"))
  foo3 <- subset(x, factor == met1)
  foo4 <- subset(x, factor == met2)
  
  # get coefficient of determination (Rsquared)
  Rsqr1 <- summary(lm(foo1$obs.data ~ foo1$predicted.data))$r.squared
  Rsqr2 <- summary(lm(foo2$obs.data ~ foo2$predicted.data))$r.squared
  Rsqr3 <- summary(lm(foo3$obs.data ~ foo3$predicted.data))$r.squared
  Rsqr4 <- summary(lm(foo4$obs.data ~ foo4$predicted.data))$r.squared
  p1 <- lmp(lm(foo1$obs.data ~ foo1$predicted.data)) 
  p2 <- lmp(lm(foo2$obs.data ~ foo2$predicted.data)) 
  p3 <- lmp(lm(foo3$obs.data ~ foo3$predicted.data)) 
  p4 <- lmp(lm(foo4$obs.data ~ foo4$predicted.data)) 
  
  if (n == 1){
    results <- data.frame(obs.metric = met1, pred.metric = met2, Rsquared = Rsqr1, p.value = p1, stringsAsFactors = F)
    results <- rbind(results, data.frame(obs.metric = met2, pred.metric = met1, Rsquared = Rsqr2, p.value = p2, stringsAsFactors = F))
    results <- rbind(results, data.frame(obs.metric = met1, pred.metric = met1, Rsquared = Rsqr3, p.value = p3, stringsAsFactors = F))
    results <- rbind(results, data.frame(obs.metric = met2, pred.metric = met2, Rsquared = Rsqr4, p.value = p4, stringsAsFactors = F))
    
  } else {
    results <- rbind(results, data.frame(obs.metric = met1, pred.metric = met2, Rsquared = Rsqr1, p.value = p1, stringsAsFactors = F))
    results <- rbind(results, data.frame(obs.metric = met2, pred.metric = met1, Rsquared = Rsqr2, p.value = p2, stringsAsFactors = F))
    results <- rbind(results, data.frame(obs.metric = met1, pred.metric = met1, Rsquared = Rsqr3, p.value = p3, stringsAsFactors = F))
    results <- rbind(results, data.frame(obs.metric = met2, pred.metric = met2, Rsquared = Rsqr4, p.value = p4, stringsAsFactors = F))
  }
}

results <- unique(results)

#saveRDS(results, file = "data/non.specific.Rsquared.rds")

## How many of these are better than the correctly paired model?
for (metric in unique(results$obs.metric)){
  foo <- subset(results, obs.metric == metric)  
  real <- foo[which(foo$obs.metric == foo$pred.metric),"Rsquared"]
  
  if (nrow(subset(foo, Rsquared > real)) > 0){
    print(subset(foo, Rsquared > real))
  } 
}

results$Rsquared <- round(results$Rsquared, 3)
results$p.value <- round(results$p.value, 4)
results <- results[rev(order(results$Rsquared)),]
write.csv(results, file = "figures/non.specific.csv", row.names = F, quote = F)
