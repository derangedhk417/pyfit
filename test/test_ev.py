# Author: Adam Robinson
# This file runs pyfit without performing any actual training and has it
# export energy vs. volume curves. It compares this data to reference files
# computed using the old code. If this test passes, it means that the program
# produces the same energy output for the same neural network and input
# dataset.

import subprocess
import os
import sys
import numpy as np
import code

# This should remove any complications related to executing the
# program from this working directory.
from sys import path
path.append("../src")

def float_6(f):
	# Parses the floating point number, represented as a string,
	# into a float, but as if it was rounded to a certain digit
	# when it was a string.
	return float('%.6E'%float(f))

def compare(fnew, fold):
	# First, parse the new file format.
	# The first line should be the volumes and the second
	# line should be the energies.

	with open(fnew, 'r') as file:
		raw_new = file.read()

	with open(fold, 'r') as file:
		raw_old = file.read()

	new_lines = raw_new.rstrip().split('\n')
	new_v     = [float(i) for i in new_lines[0].split(' ')[1:]]
	new_e     = [float(i) for i in new_lines[1].split(' ')[1:]]

	old_lines = raw_old.rstrip().split('\n')[2:]
	old_cells = []

	for line in old_lines:
		cells = line.split(' ')
		old_cells.append([i for i in cells if i != '' and not i.isspace()])

	# When all of the data is used for training (no validation). pyfit will
	# order it first by group name, then by structure volume. We need to do
	# the same in order to make a comparison.
	old = [(i[0], float_6(i[2]), float_6(i[4])) for i in old_cells]
	old = sorted(old, key=lambda x: (x[0], x[1]))

	old_v = [i[1] for i in old]
	old_e = [i[2] for i in old]

	# Now that they are loaded, compare them and print any that are 
	# not equal.

	for l, r in zip(new_e, old_e):
		diff = np.abs((l - r) / r)
		if diff > 1e-5:
			print('%+.6E - %+.6E = %+.6E'%(l, r, l - r))
			return False

	return True

if __name__ == '__main__':
	nn_names      = ['nn/%02i.nn.dat'%i       for i in range(10)]
	lsparam_names = ['ref/%02i.lsparam.dat'%i for i in range(10)]
	ev_names      = ['ref/%02i.ev.dat'%i      for i in range(10)]

	# For each file in the list of lsparam_names, run pyfit, set to
	# generate an lsparam file. After creating each file, load the
	# corresponding ref file and compare it. Do the comparision before
	# running the next generation script, in order to catch errors 
	# before wasting too much time.

	for lsp, ev, nn in zip(lsparam_names, ev_names, nn_names):
		run_str  = 'python3 ../src/pyfit.py --config-file=ev_config.json '
		run_str += '--run-training '
		run_str += '--training-set-in=%s '
		run_str += '--network-input-file=%s'
		run_str %= (lsp, nn)

		print(run_str)
		output = subprocess.getoutput(run_str)

		if os.path.isfile('tmp.nn.dat'):
			os.remove('tmp.nn.dat')

			# Compare the ev file from the old code to the one
			# that was just produced.
			if not compare('ev.txt', ev):
				print("Comparison failed, exiting . . . ")
				break
		else:
			print("pyfit failed to write the nn file")
			print("stdout:\n\n")
			print(output)
			break