# This file serves to process command line arguments, load the configuration 
# file and call the appropriate functionality. The majority of the intensive
# calculations are done in other files.

import os
import sys
import copy
import torch
import numpy as np

from args import ParseArgs, ValidateArgs, PrintHelp

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

	if status != 0:
		return status



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
	