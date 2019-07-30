#!/usr/bin/env python3

# Author: Adam Robinson
# This script is helpful for producing good quality graphs of the properties 
# of a trained neural network. It only tries to generate the actual graph
# contents. Axis labels and titles are meant to be done elsewhere. I usually
# use keynote to accomplish this.

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

def xyz_to_img(x, y, z, size):

	sigmax = (x_rng[-1] - x_rng[0])*sigmax
	sigmay = (y_rng[-1] - y_rng[0])*sigmay

	x = np.array(x)
	y = np.array(y)
	z = np.array(z)

	grid = np.zeros((size, size))

	for yidx, yi in enumerate(y.shape[0]):
		for xidx, xi in enumerate(x.shape[0]):
			weights  = (1 / (2*np.pi*sigmax*sigmay))*np.exp(-(((x - xi)**2 / (2*sigmax**2)) + ((y - yi)**2 / (2*sigmay**2))))
			mean_val = np.average(z, weights=weights)
			grid[yidx][xidx] = mean_val

	return grid

def format_axis(ax):
	ax.xaxis.set_tick_params(width=2)
	ax.yaxis.set_tick_params(width=2)
	
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(20) 
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(20) 
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(1.8)

c_list = [
	'green', 'cyan', 'blue', 
	'orange', 'yellow',
	'magenta'
]
def volume_energy_plots(full_structures, nn_energies, filters, title):
	full       = full_structures
	names      = [struct[0].group_name for struct in full]
	vol        = [struct[0].structure_volume for struct in full]
	n_atom     = [struct[0].structure_n_atoms for struct in full]
	energy     = [e for e in nn_energies.detach().numpy()]
	dft_energy = [struct[0].structure_energy for struct in full]
	combined_array = list(zip(
		names, vol, n_atom, 
		energy, dft_energy
	))

	# Now we split this up into a separate set of data points for
	# each group that the user wants plotted.

	sets = {}
	for name in filters:
		current = []
		if '*' in name:
			for item in combined_array:
				if len(fnmatch.filter([item[0]], name)) != 0:
					current.append(item)
		else:
			for item in combined_array:
				if item[0] == name:
					current.append(item)
		sets[name] = current

	# Now we actually create a series in a plot for each group.
	fig, ax = plt.subplots(1, 1)
	plots   = []
	labels  = []
	for i, k in enumerate(sets):
		# sort by volume
		items = sorted(sets[k], key=lambda x: x[1])

		# create a volume vs. energy per atom series
		v     = [i[1] for i in items]
		e_nn  = [i[3] / i[2] for i in items]
		e_dft = [i[4] / i[2] for i in items]

		
		pl0 = ax.scatter(v, e_nn, s=50, color='red', marker='1')
		pl1 = ax.scatter(
			v, e_dft, marker='o', edgecolors=c_list[i],
			s=20, facecolors='none')

		plots.append(pl1)
		labels.append(k)
	ax.legend(plots, labels)
	format_axis(ax)
	ax.set_title(title)
	plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Helps plot the final properties of a trained neural ' +
		'network.',
	)

	parser.add_argument(
		'-n', '--neural-network', dest='network_file', type=str, required=True, 
		help='The neural network to evaluate.'
	)

	parser.add_argument(
		'-t', '--training-set', dest='training_file', type=str, required=True, 
		help='The training set file to use for comparisons.'
	)

	parser.add_argument(
		'-s', '--validation-seed', dest='seed', type=int, default=0,
		help='The seed to use when splitting the training set. 0 means use ' +
		'probabilistic behavior.'
	)

	parser.add_argument(
		'-r', '--validation-ratio', dest='ratio', type=float, default=1.0,
		help='The ratio of training data to all data.'
	)

	parser.add_argument(
		'-H', '--histograms', dest='histograms', action='store_true',
		help='Construct histograms.'
	)

	parser.add_argument(
		'-P', '--parity', dest='parity', action='store_true',
		help='Construct parity plots.'
	)

	parser.add_argument(
		'-V', '--volume-vs-energy', dest='volume_v_energy', nargs='*', type=str,
		metavar='GROUP_NAME', default=[],
		help='Construct energy vs. volume plots for the following groups.'
	)

	args = parser.parse_args()

	# Load the training set and neural network.
	training_set = TrainingSet().loadFromFile(args.training_file)
	potential    = NetworkPotential().loadFromFile(args.network_file)

	# Make sure they actually match up.
	if potential.config != training_set.config:
		print("Training set and Potential configurations do not match.")
		exit(1)

	# Split the training data into training and validation sets.
	dataset = TorchTrainingData(
		training_set,
		args.ratio,
		args.seed
	)

	# Construct a neural network that is ready to be evaluated.
	nn = TorchNetwork(
		potential, 
		dataset.train_reduction,
		0.0
	)

	# Evaluate it for training data.
	train_energies = nn(dataset.train_lsp)

	# Calculate the per-structure error.
	train_diff        = train_energies - dataset.train_energies
	train_diff_scaled = torch.mul(train_diff, dataset.train_reciprocals)
	train_diff_scaled = train_diff_scaled.detach().numpy()


	# Evaluate it for validation data.
	nn.setReductionMatrix(dataset.val_reduction)
	val_energies = nn(dataset.val_lsp)
	nn.setReductionMatrix(dataset.train_reduction)

	val_diff        = val_energies - dataset.val_energies
	val_diff_scaled = torch.mul(val_diff, dataset.val_reciprocals)
	val_diff_scaled = val_diff_scaled.detach().numpy()

	# Print out the 20 worst structural groups and their validation
	# error.
	val_structures = dataset.full_validation_structures
	names          = [struct[0].group_name for struct in val_structures]
	ids            = [struct[0].structure_id for struct in val_structures]
	combined_array = []
	for error, name, struct_id in zip(val_diff_scaled, names, ids):
		combined_array.append((error, name, struct_id))

	combined_array = sorted(combined_array, key=lambda x: np.abs(x[0]))
	print("The Worst 20 Structures by Validation Error:")
	for i in combined_array[-20:]:
		print('    %8s (id: %04i) : %+5.4f'%(i[1], i[2], i[0]))

	# Construct a histogram for the training data and a histogram for 
	# the validation data.
	if args.histograms:
		fig, ax = plt.subplots(1, 1)
		ax.hist(
			train_diff_scaled*1000, 
			45, facecolor='orange', edgecolor='black',
			density=True
		)
		ax.set_title('training')
		format_axis(ax)
		ax.yaxis.set_major_formatter(ticker.FuncFormatter(
			lambda x, pos: str(int(x * 100))
		))
		plt.show()

		fig, ax = plt.subplots(1, 1)
		ax.hist(
			val_diff_scaled*1000, 
			45, facecolor='orange', edgecolor='black',
			density=True
		)
		ax.set_title('validation')
		format_axis(ax)
		ax.yaxis.set_major_formatter(ticker.FuncFormatter(
			lambda x, pos: str(int(x * 100))
		))
		plt.show()

	if args.parity:

		# Construct parity plots of the predicted energy vs the dft energy.

		fig, ax  = plt.subplots(1, 1)
		line_rng = np.arange(
			dataset.train_energies.min(), 
			dataset.train_energies.max()
		)
		ax.plot(line_rng, line_rng, color='magenta')

		ax.scatter(
			dataset.train_energies, train_energies.detach().numpy(),
			s=10, zorder=1000000
		)
		ax.set_title('training')
		format_axis(ax)
		plt.show()

		fig, ax = plt.subplots(1, 1)
		line_rng = np.arange(
			dataset.val_energies.min(), 
			dataset.val_energies.max()
		)
		ax.plot(line_rng, line_rng, color='magenta')
		ax.scatter(
			dataset.val_energies, val_energies.detach().numpy(),
			s=10, zorder=1000000
		)
		ax.set_title('validation')
		format_axis(ax)
		plt.show()

	if args.volume_v_energy != []:
		# For both the training set and the validation set, we need
		# an array that pairs the per-atom volume, the per-atom energy
		# as predicted by dft, the per-atom energy as predicted by 
		# the nn and the name of the group.
		volume_energy_plots(
			dataset.full_validation_structures,
			val_energies,
			args.volume_v_energy,
			'validation'
		)

		volume_energy_plots(
			dataset.full_training_structures,
			train_energies,
			args.volume_v_energy,
			'training'
		)