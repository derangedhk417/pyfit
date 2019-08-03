#!/usr/bin/env python3

# Author: Adam Robinson
# This program will determine the mean density of points in feature space
# for each structure in the training data. It will then attempt to select
# the training-validation split that maximizes the mean density of points
# around the validation data points.

import code
import argparse
import os
import time
import sys
import torch
import json
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import fnmatch
from   scipy import interpolate

from sys import path
path.append(sys.path[0] + '/../src')
from potential    import NetworkPotential
from training_set import TrainingSet
from train        import TorchTrainingData, TorchNetwork
from lsp          import computeParameters
from util         import ProgressBar
from fnmatch      import filter     as match


def get_args():
	parser = argparse.ArgumentParser(
		description='Constructs a training-validation split that maximizes ' +
		'the data point density in feature space around all validation inputs ',
	)

	parser.add_argument(
		'-t', '--training-set', dest='training_file', type=str, required=True, 
		help='The training set file that was used to train the networks.'
	)

	parser.add_argument(
		'-r', '--validation-ratio', dest='ratio', type=float, default=0.9,
		help='The ratio of training data to all data (default 0.9)'
	)

	parser.add_argument(
		'-g', '--sigma', dest='sigma', type=float, 
		default=0.05,
		help='The sigma value for the gaussians, as a percentage of the ' +
		'of the values on each dimension.'
	)

	parser.add_argument(
		'-o', '--out-file', dest='out_file', type=str, default='',
		help='The file to dump the results into.'
	)

	parser.add_argument(
		'--cpu', dest='force_cpu', action='store_true', 
		help='Force operations to take place on the CPU.'
	)

	return parser.parse_args()

def score(validation, training, sigma, device, seed):
	# First, we calculate the sigma values. We will do this over
	# the entire range, for both validation and training.
	validation = validation
	training   = training

	n_dims   = validation.shape[1]
	n_points = training.shape[0]

	real_sigma = []
	for s in range(n_dims):
		# Get the min and max value in this dimension.
		v = validation[:, s]
		t = training[:, s]

		min_v = v.min()
		min_t = t.min()

		max_v = v.max()
		max_t = t.max()

		_min = min(min_v, min_t)
		_max = max(max_v, max_t)

		real_sigma.append(sigma * (_max - _min))

	real_sigma = torch.tensor(real_sigma).to(device)

	densities = torch.zeros((validation.shape[0])).to(device)

	progress = ProgressBar(
		"Calculating (%05i)"%seed, 22, 
		validation.shape[0], update_every=25
	)

	# Now that we know the appropriate sigma value for each
	# dimension, we can actually compute the gaussians.
	for i, point in enumerate(validation):
		diffs        = point - training
		frac         = diffs / real_sigma
		inner_term   = 0.5 * (frac**2)
		inner_term   = -inner_term.sum(dim=1)
		gaussians    = torch.exp(inner_term)
		gaussian     = gaussians.sum()
		densities[i] = gaussian / n_points

		progress.update(i + 1)

	progress.finish()

	# Now we have a density for each point in the validation 
	# data. Normalize the values.

	# densities  = densities - densities.min()
	# densities /= densities.max() - densities.min()

	# Return statistics.

	return {
		'min' : densities.min().cpu().item(),
		'max' : densities.max().cpu().item(),
		'avg' : densities.mean().cpu().item(),
		'std' : densities.std().cpu().item()
	}


if __name__ == '__main__':
	
	args = get_args()

	# Load the training set.
	training_set = TrainingSet().loadFromFile(args.training_file)
	stats        = {}

	if args.force_cpu or not torch.cuda.is_available():
		device = 'cpu'
	else:
		device = 'cuda:0'

	# Before we can start setting up the training set, we need
	# to build a training set structure where everything is
	# considered training data.
	dataset = TorchTrainingData(
		training_set,
		1.0,
		0
	).to(device)

	# Now we need to score each structure in the training data as 
	# if it were validation data and the rest of the data remained
	# training data.

	structures = dataset.full_training_structures

	

	try:
		for seed in range(args.seed_range[0], args.seed_range[1] + 1):
			# Split the training data into training and validation sets.
			

			# Run the score function.
			density_stats = score(
				dataset.val_lsp,
				dataset.train_lsp,
				args.sigma,
				device,
				seed
			)

			stats[seed] = density_stats
	except KeyboardInterrupt:
		print("SIGINT Detected, cleaning up . . . ")

	if args.out_file != '':
		with open(args.out_file, 'w') as file:
			file.write(json.dumps(stats))
	else:
		print(stats)