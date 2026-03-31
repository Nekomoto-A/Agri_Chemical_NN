#!/home/roli/anaconda3/bin/python
import sys, os, re, numpy as np
from subprocess import call
from multiprocessing import Pool

## User Input
threads = 10
number_of_seeds = 100

def spawn_ML(seed):
	os.system(' '.join([
		"Rscript runML.R",
		str(seed)
	]))


# determine number of batches of seeds to run per the thread count
splits = int(number_of_seeds/threads)

# run through each batch of seeds
for batch in np.array_split(range(number_of_seeds), splits):

	# spawn a thread for each seed number
	with Pool(processes=threads) as pool:
		result = pool.map(spawn_ML, batch.tolist())
