# This file serves to process command line arguments, load the configuration 
# file and call the appropriate functionality. The majority of the intensive
# calculations are done in other files.

import os
import sys
import copy
import torch
import numpy as np

if __name__ == '__main__':
	# The first task is to parse the arguments into a structure.
	# This structure will then be used to modify the information in the 
	# configuration file.