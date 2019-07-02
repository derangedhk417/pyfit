# This file serves to process command line arguments, load the configuration 
# file and call the appropriate functionality. The majority of the intensive
# calculations are done in other files.

import os
import sys
import copy
import torch
import numpy as np

from args   import ParseArgs, PrintHelp
from config import ConstructConfiguration

if __name__ == '__main__':
	# The first task is to parse the arguments into a structure.
	# This structure will then be used to modify the information in the 
	# configuration file.

	# Parse the arguments.
	arguments, printed_help = ParseArgs(sys.argv)

	if printed_help:
		exit()

	# The arguments are valid at face values. Check the additional args against
	# the specification in config.py. If everything is valid, construct the 
	# configuration for this run of the program.

	configuration, success = ConstructConfiguration(arguments)

	if not success:
		PrintHelp()
		exit()