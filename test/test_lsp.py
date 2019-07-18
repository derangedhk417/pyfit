# Author: Adam Robinson
# This file uses the ten neural network potential files in the nn directory
# to generate a set of lsparam files. It then compares them to the ten files
# in the ref folder. These files were generated using the previous lsp 
# generating code, written in c. They are assumed to be correct.

import subprocess
import os
import numpy as np

# This should remove any complications related to executing the
# program from this working directory.
from sys import path
path.append("../src")

from training_set import TrainingSet

# These are all maximum tolerable error statistics in percent.
max_err_threshold  = 0.01
mean_err_threshold = 0.00001
std_err_threshold  = 0.00001

def compareInputs(new, old, i, j):
	def location():
		msg  = "structure: %i, input: %i"
		msg %= (i, j)
		return msg

	def err(a, b):
		print(print(location()))
		print('%s != %s'%(a, b))

	if new.group_name != old.group_name:
		err(new.group_name, old.group_name)
		return False

	if new.group_id != old.group_id:
		err(new.group_id, old.group_id)
		return False

	if new.structure_id != old.structure_id:
		err(new.structure_id, old.structure_id)
		return False

	if new.structure_n_atoms != old.structure_n_atoms:
		err(new.structure_n_atoms, old.structure_n_atoms)
		return False

	if new.structure_energy != old.structure_energy:
		err(new.structure_energy, old.structure_energy)
		return False

	if new.structure_volume != old.structure_volume:
		err(new.structure_volume, old.structure_volume)
		return False

def compareTrainingSets(new, old):
	new_set = TrainingSet().loadFromFile(new)
	old_set = TrainingSet().loadFromFile(old)

	# First, compare all of the structural parameters.
	for i, struct in enumerate(new_set.structures):
		for j, new_in in enumerate(struct):
			# Firstly, make sure that the same exact item exists in 
			# the old training set.
			def location():
				msg  = "structure: %i, input: %i"
				msg %= (i, j)
				return msg

			try:
				old_in = old_set.structures[i][j]
			except:
				print(location() + ' missing.')
				return False

			# Compare all members between the two and summarize differences if
			# they are found.
			if not compareInputs(new_in, old_in, i, j):
				return False

			# Now we get the error in the structure parameters.
			percent_errors  = new_in.structure_params
			percent_errors -= old_in.structure_params
			percent_errors /= old_in.structure_params
			percent_errors  = np.abs(percent_errors) * 100

			_max = percent_errors.max()
			_avg = percent_errors.mean()
			_std = percent_errors.std()

			err = False
			if _max > max_err_threshold:
				print("structural parameter error exceeded max threshold.")
				err = True

			if _avg > mean_err_threshold:
				print("structural parameter error exceeded mean threshold.")
				err = True

			if _std > std_err_threshold:
				print("structural parameter error exceeded std threshold.")
				err = True
				

			if err:
				new_str = ' '.join([str(i) for i in new_in.structure_params])
				print('\tnew: %s'%(new_str))

				old_str = ' '.join([str(i) for i in old_in.structure_params])
				print('\told: %s'%(old_str))

				return False


if __name__ == '__main__':
	lsparam_names = ['out/%02i.lsparam.nogit.dat'%i for i in range(10)]
	nn_names      = ['nn/%02i.nn.dat'%i   for i in range(10)]
	ref_names     = ['ref/%02i.lsparam.dat'%i for i in range(10)]
	set_name      = 'input/ab.poscar.dat'

	# Make sure that none of the output files already exist.
	for lsp in lsparam_names:
		if os.path.isfile(lsp):
			print('%s already exists, deleting . . . '%lsp)
			os.remove(lsp)

	# For each file in the list of lsparam_names, run pyfit, set to
	# generate an lsparam file. After creating each file, load the
	# corresponding ref file and compare it. Do the comparision before
	# running the next generation script, in order to catch errors 
	# before wasting too much time.

	for lsp, ref, nn in zip(lsparam_names, ref_names, nn_names):
		run_str  = 'python3 ../src/pyfit.py --config-file=test_config.json '
		run_str += '--generate-training-set --dft-file=%s '
		run_str += '--training-set-out=%s '
		run_str += '--network-input-file=%s'
		run_str %= (set_name, lsp, nn)

		print(run_str)
		output = subprocess.getoutput(run_str)

		if os.path.isfile(lsp):
			# Load the file and compare it to the ref file.
			if not compareTrainingSets(lsp, ref):
				print("Comparison failed, exiting . . . ")
				break
		else:
			print("pyfit failed to write the lsparam file")
			print("stdout:\n\n")
			print(output)
			break