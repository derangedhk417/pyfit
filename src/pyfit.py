#!/usr/bin/env python3
# Authors: Adam Robinson, James Hickman
# This file serves to process command line arguments, load the configuration 
# file and call the appropriate functionality. The majority of the intensive
# calculations are done in other files.

import os
import sys
import copy
import numpy as np

from args         import ParseArgs, ValidateArgs, PrintHelp
from config       import PotentialConfig
from poscar       import PoscarLoader
from potential    import NetworkPotential
from training_set import TrainingSet
from neighbor     import GenerateNeighborList
from lsp          import GenerateLocalStructureParams
from train        import Trainer


def RunPyfit(config):
	# Try to ensure that all of the configuration settings make sense. If the 
	# run is doomed to fail we want to catch is now and not when the job has 
	# been running on a cluster for ten minutes already.

	# This function will do a decent job of pre-validating everything.
	# It tries to print helpful error information and will print the help
	# documentation when appropriate. If execution continues after this, it
	# is safe to say that the configuration is at least somewhat sane. It is
	# still possible that a file has invalid contents though.
	status = ValidateArgs(config)

	if not isinstance(status, tuple):
		return status

	log = status[1]

	# Now that basic configuration stuff is out of the way, we need to 
	# generate a training set, train a neural network or both. 

	potential    = None
	training_set = None

	force_training = config.force_interval > 0

	if config.generate_training_set:
		poscar_data = PoscarLoader(
			config.e_shift, 
			log=log, 
			has_force=force_training
		)
		poscar_data = poscar_data.loadFromFile(config.dft_input_file)

		potential   = NetworkPotential(log=log)
		potential   = potential.loadFromFile(config.neural_network_in)

		neighborLists = GenerateNeighborList(
			poscar_data.structures,
			potential,
			log=log
		)

		lsps = GenerateLocalStructureParams(
			neighborLists,
			potential.config,
			log=log
		)

		training_set = TrainingSet(log=log).loadFromMemory(
			poscar_data,
			lsps,
			potential,
		)

		training_set.writeToFile(config.training_set_output_file)

	if config.run_training:
		# If we generated the training set in this run, then there was 
		# already a validation check run to make sure that the generated
		# training output file matches the training input file being used.
		# Instead of loading from disk, just use the existing instance.
		if not config.generate_training_set:
			training_set = TrainingSet(log=log, has_force=force_training)
			training_set = training_set.loadFromFile(config.training_set_in)
			potential    = NetworkPotential(log=log)
			potential    = potential.loadFromFile(config.neural_network_in)

			if potential.config != training_set.config:
				msg  = "The training set file and network potential file have "
				msg += "different configurations. Please check them."
				print(msg)
				log.log("potential.config != training_set.config")
				return 1

		if config.randomize:
			log.log("Randomizing network potential parameters.")
			potential.randomizeNetwork()
			potential.loadNetwork(None, values_loaded=True)

		if not config.no_warn:
			if not training_set.generateWarnings(config.validation_ratio):
				return 1

		kwargs = {}
		if force_training:
			kwargs['neighborList'] = neighborLists

		kwargs['log'] = log

		# By this point, 'training_set' holds a training set instance, one way
		# or another. Now we actually run the training.
		trainer = Trainer(potential, training_set, config, **kwargs)

		trainer.train()


	

if __name__ == '__main__':
	# Parse the arguments. And construct a configuration structure that can be
	# passed around to the functions in the program.
	config = ParseArgs(sys.argv)
	# The program is structured this way so that a script that automates a run 
	# of pyfit can do so programmatically in a very straightforward manner. A
	# user could have code like the following, if they wanted to automate 
	# multiple runs.
	# import pyfit
	# pyfit.RunPyfit(my_arg_structure)
	RunPyfit(config)