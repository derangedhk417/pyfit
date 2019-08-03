#!/usr/bin/env python3

# Author: Adam Robinson
# This is designed to analyze one or more training-validation splits.
# It tries to quantify the degree to which the validation data points
# are an interpolation between the training data points. See select.pdf
# for details on how this is implemented.

import code
import argparse
import os
import time
import sys
import torch
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
		description='Helps you construct training-validation splits that ' +
		'are a good interpolation over the training set.',
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
		'-s', '--seed-range', dest='seed_range', type=int, 
		default=[1, 10], nargs=2, metavar=('LOW', 'HIGH'),
		help='The range of seed values to try for the split.'
	)

	parser.add_argument(
		'-g', '--sigma', dest='sigma', type=float, 
		default=0.05,
		help='The sigma value for the gaussians, as a percentage of the ' +
		'of the values on each dimension.'
	)

	return parser.parse_args()

def score(validation, training, sigma):
	# First, we calculate the sigma values. We will do this over
	# the entire range, for both validation and training.
	validation = validation
	training   = training

	n_dims = validation.shape[1]

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

	real_sigma = torch.tensor(real_sigma)

	densities = torch.zeros((validation.shape[0]))

	progress = ProgressBar(
		"Calculating ", 22, 
		validation.shape[0], update_every=25
	)

	# Now that we know the appropriate sigma value for each
	# dimension, we can actually compute the gaussians.
	for i, point in enumerate(validation):
		diffs        = point - training
		frac         = diffs / real_sigma
		inner_term   = 0.5 * (frac**2)
		inner_term   = -inner_term.sum(dim=1)
		gaussians    = np.exp(inner_term)
		gaussian     = gaussians.sum()
		densities[i] = gaussian

		progress.update(i + 1)

	progress.finish()

	# Now we have a density for each point in the validation 
	# data. Normalize the values.

	# densities  = densities - densities.min()
	# densities /= densities.max() - densities.min()

	# Return statistics.

	return {
		'min' : densities.min().item(),
		'max' : densities.max().item(),
		'avg' : densities.mean().item(),
		'std' : densities.std().item()
	}


if __name__ == '__main__':
	
	args = get_args()

	# Load the training set.
	training_set = TrainingSet().loadFromFile(args.training_file)
	
	stats = []

	for seed in range(args.seed_range[0], args.seed_range[1] + 1):
		# Split the training data into training and validation sets.
		dataset = TorchTrainingData(
			training_set,
			args.ratio,
			seed
		)

		# Run the score function.
		density_stats = score(
			dataset.val_lsp,
			dataset.train_lsp,
			args.sigma
		)

		stats.append(density_stats)

	print(stats)