#!/usr/bin/env python3

# Author: Adam Robinson
# This script is designed to help you remove problem structures from a
# training set. It is usually wise to use outliers.py to identify bad
# structures and then remove them from the poscar file with this script.
# You can then generate a new training set file.

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
from poscar       import PoscarLoader
from train        import TorchTrainingData, TorchNetwork
from lsp          import computeParameters
from fnmatch      import filter     as match


def get_args():
	parser = argparse.ArgumentParser(
		description='Helps you find outlier remove outlier DFT data once ' +
		'you\'ve identified it (usually with outliers.py).',
	)

	parser.add_argument(
		'-d', '--dft-file', dest='dft_file', type=str, required=True, 
		help='The dft file to remove structures from.'
	)

	parser.add_argument(
		'-s', '--structure-ids', dest='structure_ids', type=int, nargs='*',
		default=[], metavar='ID', 
		help='The ids of the structures to remove.'
	)


	parser.add_argument(
		'-o', '--output-file', dest='output_file', type=str, required=True, 
		help='Where to write the resulting file.'
	)


	return parser.parse_args()

if __name__ == '__main__':
	
	args   = get_args()
	poscar = PoscarLoader(0.0).loadFromFile(args.dft_file)
	

	# pyfit assigns ids to structures sequentially starting from zero in the
	# order that it finds them in the poscar file. We just need to write the
	# poscar file back out, omitting the structures that the user wants 
	# skipped.
	with open(args.output_file, 'w', 10000) as file:
		for idx, struct in enumerate(poscar):
			if idx not in args.structure_ids:
				file.write(str(struct))

	print("Done")