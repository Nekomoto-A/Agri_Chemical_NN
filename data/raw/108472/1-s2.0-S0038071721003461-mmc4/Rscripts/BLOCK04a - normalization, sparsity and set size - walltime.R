library(plyr)
library(Hmisc)
library(ggplot2)

# Import
x <- read.table(file = "../models/ML.final.time.tsv", sep="\t", header=T, stringsAsFactors = F)
x$run <- unlist(lapply(x$experiment, function(x) unlist(strsplit(x, "\\."))[2]))
x$experiment <- unlist(lapply(x$experiment, function(x) unlist(strsplit(x, "\\."))[1]))

# Convert to hrs
x$hrs <- x$time/3600
x$min <- x$time/60

# fix trailing "_" in naming
x$factor <- gsub("_$","",x$factor)

# Report averages
averages <- ddply(x, ~ model + rank, summarise, avg.hrs = mean(hrs))
averages <- averages[order(averages$model, averages$rank),]
averages$avg.hrs <- round(averages$avg.hrs, 2)
write.csv(reshape(averages, direction = "wide", idvar = "rank", timevar = "model"), file = "figures/walltime.csv", quote =F, row.names = F)

# Correlate train time with feature set size
for (mod in c("L2LinearSVM", "RandomForest")){
  if (mod == "RandomForest"){
    averages <- ddply(subset(x, model == mod), ~ rank + dataset, summarise, avg = mean(hrs))
    lab = "Time (hrs)"
  } else {
    averages <- ddply(subset(x, model == mod), ~ rank + dataset, summarise, avg = mean(min))
    lab = "Time (min)"
  }
  set <- read.csv(file = "../models/feature.set.size.csv", header = T, stringsAsFactors = F)
  averages <- merge(averages, set, by = c("dataset","rank"))
  
  # Plot Dataset
  plot <- ggplot(averages, aes(x = size, y = avg)) + 
    geom_point() +
    stat_smooth(method = "lm") + ylab(lab)
  
  print(plot)
  print(rcorr(averages$avg, averages$size))
}

#for (exp in c("exp1","exp2")){
#  x$rank <- factor(x$rank, levels = rev(c("Order","Family","Genus","ASV")))
#  plot <- ggplot(subset(x, time.period == "to complete" & experiment == exp), aes(x=factor, y=time, colour = rank)) + geom_boxplot(outlier.shape = NA) + ylab("Time (hrs)") + facet_grid(~model+dataset) + ggtitle(exp)
#  plot <- plot + geom_point(position=position_jitterdodge(jitter.width = 0.1), alpha=0.5, size = 0.5)
#  print(plot)
#  ggsave(plot, filename=paste('figures/walltime.',exp,'.boxplot.pdf',sep=""), height=20, width=30)
#}

