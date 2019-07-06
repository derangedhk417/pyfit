# This file serves to process command line arguments, load the configuration 
# file and call the appropriate functionality. The majority of the intensive
# calculations are done in other files.

import os
import sys
import copy
import torch
import numpy as np

from args import ParseArgs, PrintHelp

if __name__ == '__main__':
	# The first task is to parse the arguments into a structure.
	# This structure will then be used to modify the information in the 
	# configuration file.

	# Parse the arguments. And construct a configuration structure that can be
	# passed around to the functions in the program.
	config = ParseArgs(sys.argv)

	print("Verbosity:                 %i"%config.verbosity)
	print("Log Path:                  %s"%config.log_path)
	print("E Shift:                   %f"%config.e_shift)
	print("Unweighted Negative Error: %i"%config.unweighted_negative_error)