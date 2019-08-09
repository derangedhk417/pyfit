#!/usr/bin/env python3

# Author: Adam Robinson
# This script is meant to aid in the process of analyzing a large number of
# trained potentials. It allows you to determine whether or not there are 
# any groups or structures that consistently have poor validation (or training)
# error across multiple initial conditions. This script is based on the assumption
# that you have multiple initial conditions, each with one or more training runs. 

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
from fnmatch      import filter     as match


def get_args():
	parser = argparse.ArgumentParser(
		description='Helps you find outlier DFT data that neural networks ' +
		'have a hard time fitting.',
	)

	parser.add_argument(
		'-d', '--network-directory', dest='network_dir', type=str, required=True, 
		help='The directory to search in for neural network files.'
	)

	parser.add_argument(
		'-n', '--network-pattern', dest='network_pattern', type=str, required=True, 
		help='The pattern to use to find network files to test.'
	)

	parser.add_argument(
		'-t', '--training-set', dest='training_file', type=str, required=True, 
		help='The training set file that was used to train the networks.'
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
		'-b', '--error-threshold', dest='error_threshold', type=float, default=0.010,
		help='All structures with an error worse than this will be compared.'
	)

	parser.add_argument(
		'-V', '--use-validation', dest='use_val', action='store_true',
		help='Compare validation error, default is training error.'
	)

	parser.add_argument(
		'-E', '--sort-by-error', dest='sort_by_error', action='store_true',
		help='Sort the results by the average error. Default is to sort by ' +
		'the count.'
	)

	return parser.parse_args()

# Recursively develops a list of all objects in the given directory.
def _all(_dir):
	result = []
	for dp, dn, fn in os.walk(os.path.expanduser(_dir)):
		for f in fn:
			result.append(os.path.join(dp, f))
	return result

if __name__ == '__main__':
	
	args = get_args()

	# Load the training set.
	training_set        = TrainingSet().loadFromFile(args.training_file)
	all_potential_files = match(_all(args.network_dir), args.network_pattern)
	all_potentials      = []
	all_networks        = []

	print("Analyzing %i networks"%len(all_potential_files))

	# Search for neural network files.
	for path in all_potential_files:
		all_potentials.append(NetworkPotential().loadFromFile(path))

	# Split the training data into training and validation sets.
	dataset = TorchTrainingData(
		training_set,
		args.ratio,
		args.seed
	)



	# Construct a neural network that is ready to be evaluated for each
	# potential.
	for potential in all_potentials:
		nn = TorchNetwork(
			potential, 
			dataset.train_reduction,
			0.0
		)
		if args.use_val:
			nn.setReductionMatrix(dataset.val_reduction)
		all_networks.append(nn)

	# Depending on what the user requested, get the error in either 
	# the training or validation data. 

	correct_energies = None
	reciprocals      = None
	lsps             = None
	volumes          = None
	all_structures   = None


	if args.use_val:
		correct_energies = dataset.val_energies
		reciprocals      = dataset.val_reciprocals
		lsps             = dataset.val_lsp
		volumes          = dataset.val_volumes
		all_structures   = dataset.full_validation_structures
	else:
		correct_energies = dataset.train_energies
		reciprocals      = dataset.train_reciprocals
		lsps             = dataset.train_lsp
		volumes          = dataset.train_volumes
		all_structures   = dataset.full_training_structures
	
	all_errors = []
	for nn in all_networks:
		diff   = nn(lsps) - correct_energies
		scaled = torch.mul(diff, reciprocals)
		all_errors.append(scaled.detach().numpy().T[0])

	# We now have the per structure error for every structure and
	# for every neural network. We also need a list of structure 
	# ids and group names to associate with this data so that the
	# final report can be written in a readable format.

	all_ids    = []
	all_groups = []

	for struct in all_structures:
		all_ids.append(struct[0].structure_id)
		all_groups.append(struct[0].group_name)

	all_ids    = np.array(all_ids)
	all_groups = np.array(all_groups)
	volumes    = np.array(volumes)
	n_atoms    = 1 / reciprocals.detach().numpy().T[0]
	energies   = np.array(correct_energies.detach().numpy().T[0] / n_atoms)

	# We now have all of the necessary information loaded into
	# structures sufficient to make a comparison. We need to prepare
	# a summary of the structural groups and structure ids that 
	# most often have a high error.

	masks = []
	for error in all_errors:
		masks.append(((np.abs(np.array(error)) > args.error_threshold) & (np.abs(np.array(error)) < 2.0)))

	# We now have a list of boolean masks that can select the worst
	# values from each array. The next step is to select the worst
	# groups and the worst structure ids and make a sorted list of the
	# number of times they show up in this group of bad results.

	# fl -> filtered

	fl_errors = []
	fl_ids    = []
	fl_groups = []
	fl_vols   = []
	fl_energy = []
	fl_natom  = []
	for err, mask in zip(all_errors, masks):
		fl_errors.append(np.array(err)[mask].tolist())
		fl_ids.append(all_ids[mask].tolist())
		fl_groups.append(all_groups[mask].tolist())
		fl_vols.append(volumes[mask].tolist())
		fl_energy.append(energies[mask].tolist())
		fl_natom.append(n_atoms[mask].tolist())

	worst_by_group = {}
	worst_by_id    = {}

	for i, (err, _id, group) in enumerate(zip(fl_errors, fl_ids, fl_groups)):
		for j, (in_err, in_id, in_group) in enumerate(zip(err, _id, group)):
			if in_group not in worst_by_group:
				worst_by_group[in_group] = [1, in_err]
			else:
				worst_by_group[in_group][0] += 1
				worst_by_group[in_group][1] += in_err


			if in_id not in worst_by_id:
				worst_by_id[in_id] = [
					1, in_err, in_group,
					fl_vols[i][j], fl_energy[i][j], 
					fl_natom[i][j]
				]
			else:
				worst_by_id[in_id][0] += 1
				worst_by_id[in_id][1] += in_err

	# Now that we know the total error and the total number of
	# bad results both by group and by structure, we need to calculate
	# the averages, convert these into lists and then print them in 
	# sorted form.

	tmp = []
	for k in worst_by_group:
		count = worst_by_group[k][0]
		avg   = worst_by_group[k][1] / count

		tmp.append([k, count, avg])

	if args.sort_by_error:
		worst_by_group = sorted(tmp, key=lambda x: x[2])
	else:
		worst_by_group = sorted(tmp, key=lambda x: x[1])

	tmp = []
	for k in worst_by_id:
		count = worst_by_id[k][0]
		avg   = worst_by_id[k][1] / count
		

		tmp.append([k, count, avg, *worst_by_id[k][2:]])

	if args.sort_by_error:
		worst_by_id = sorted(tmp, key=lambda x: x[2])
	else:
		worst_by_id = sorted(tmp, key=lambda x: x[1])

	# Now we print the results in a somewhat human readable format
	# that is also fairly machine readable.

	print("By Structure ID: ")
	print("%10s %10s %10s %10s %10s %10s %10s"%('id', 'mean error', 'count', 'vol', 'e_dft', 'n atom', 'group'))

	for w in worst_by_id:
		print("%10i %10.5f %10i %10.5f %10.5f %10i %10s"%(w[0], w[2], w[1], w[4], w[5], int(w[6]), w[3]))

	print('\n\n')
	print("By Group Name: ")
	print("%15s %15s %15s"%('id', 'mean error', 'count'))

	for w in worst_by_group:
		print("%15s %15.5f %15i"%(w[0], w[2], w[1]))