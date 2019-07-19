# Authors: Adam Robinson, James Hickman
# This file contains classes and methods that assist in reading and writing
# training set files in the custom format used for neural network potentials.

import code
import numpy as np

# this parses and stores the header information
from config import PotentialConfig
from util   import ProgressBar
from copy   import deepcopy

class TrainingSet:
	def __init__(self, log=None):
		self.log = log

	def loadFromFile(self, file_path):
		if self.log is not None:
			self.log.log("Loading Training Set")
			self.log.indent()
			self.log.log("Source = disk")
			self.log.log("File   = %s"%(file_path))

		with open(file_path, 'r') as file:
			text = file.read()

		return self.loadFromText(text)

	def loadFromText(self, text):
		lines = text.rstrip().split('\n')

		self.config = PotentialConfig().loadFromText('\n'.join(lines[:8]))

		self.potential_type = int(self._getCellsFromLine(lines[8])[0])
		self.n_structures   = int(self._getCellsFromLine(lines[9])[0])
		self.n_atoms        = int(self._getCellsFromLine(lines[10])[0])

		parameters_per_atom  = self.config.n_legendre_polynomials
		parameters_per_atom *= self.config.n_r0

		progress = ProgressBar(
			"Loading Training Set", 
			22, self.n_structures, update_every = 10
		)

		# Every set of two lines from 13 onwards should correspond to a single 
		# atom. Line 12 doesn't contain useful information.

		# This code will convert the file into a list of structures. Each 
		# element in this list is a list of training inputs, each one
		# corresponding to an atom in the structure.
		self.structures = []
		idx             = 12
		current_struct  = []
		current_id      = 0
		while idx < len(lines):
			
			atom = TrainingInput().fromLines(
				lines[idx], 
				lines[idx + 1], 
				parameters_per_atom
			)

			if atom.structure_id != current_id:
				self.structures.append(current_struct)
				current_struct = []
				current_id     = atom.structure_id
				progress.update(current_id + 1)

			current_struct.append(atom)
			idx += 2

		progress.finish()
		self.structures.append(current_struct)

		if self.log is not None:
			self.log.log("Atoms      Loaded = %i"%self.n_atoms)
			self.log.log("Structures Loaded = %i"%self.n_structures)
			self.log.log("Time Elapsed = %ss"%progress.ttc)
			self.log.unindent()

		return self

	# Given a PoscarLoader instance, a list of local structure parameters
	# generated by lsp.py -> GenerateLocalStructureParams and a network
	# potential instance, will generate a training set instance. This can
	# then be immediately used, or written to a file.
	def loadFromMemory(self, poscar, lsp, potential):
		if self.log is not None:
			self.log.log("Loading Training Set")
			self.log.indent()
			self.log.log("Source = memory")

		self.config     = potential.config

		self.potential_type = 1

		self.n_structures   = poscar.n_structures
		self.n_atoms        = poscar.n_atoms

		self.structures = [] 
		
		# The PoscarLoader class itself is iterable. It will return a single
		# structure per iteration, each structure being an instance of
		# poscar.py -> PoscarStructure
		current_group_id = 1
		max_group_id     = 1
		current_group    = poscar.structures[0].comment
		group_ids        = {current_group: 1}
		for struct_idx, struct in enumerate(poscar):
			group_name = struct.comment
			if group_name != current_group:
				if group_name in group_ids:
					current_group_id = group_ids[group_name]
					current_group    = group_name
				else:
					max_group_id     += 1
					current_group_id  = max_group_id
					current_group     = group_name
					group_ids[group_name] = current_group_id

			# This is divided by two to match previous code.
			struct_volume = np.linalg.norm(
				np.dot(
					np.cross(struct.a1, struct.a2),
					struct.a3
				)
			) / struct.n_atoms

			training_input = TrainingInput()
			training_input.group_name        = group_name
			training_input.group_id          = current_group_id
			training_input.structure_id      = struct_idx 
			training_input.structure_n_atoms = struct.n_atoms
			training_input.structure_energy  = struct.energy
			training_input.structure_volume  = struct_volume

			current_struct = []
			for atom_idx in range(struct.n_atoms):
				input_copy = deepcopy(training_input)
				input_copy.structure_params = lsp[struct_idx][atom_idx]
				current_struct.append(input_copy)

			self.structures.append(current_struct)

		if self.log is not None:
			self.log.log("Atoms      Loaded = %i"%self.n_atoms)
			self.log.log("Structures Loaded = %i"%self.n_structures)
			self.log.unindent()

		return self

	# Writes the current instance to the specified file. Will overwrite any 
	# file that is already at that path.
	def writeToFile(self, file_path):
		if self.log is not None:
			self.log.log("Writing Training Set to File")
			self.log.indent()
			self.log.log("File = %s"%(file_path))

		# 50 Kb buffer because these files are always large. This should
		# make the write a little faster.
		with open(file_path, 'w', 1024*50) as file:
			file.write(self.config.toFileString(prepend_comment=True))
			file.write(' # %i - Potential Type\n'%(1))
			file.write(' # %i - Number of Structures\n'%(self.n_structures))
			file.write(' # %i - Number of Atoms\n'%(self.n_atoms))
			file.write(' # ATOM-ID GROUP-NAME GROUP_ID STRUCTURE_ID ')
			file.write('STRUCTURE_Natom STRUCTURE_E_DFT STRUCTURE_Vol\n')

			progress = ProgressBar(
				"Writing LSParams ", 
				22, self.n_atoms, update_every = 50
			)
			progress.estimate = False

			atom_idx = 0
			for struct in self.structures:
				for training_input in struct:
					file.write('ATOM-%i %s %i %i %i %.6E %.6E\n'%(
						atom_idx,
						training_input.group_name,
						training_input.group_id,
						training_input.structure_id,
						training_input.structure_n_atoms,
						training_input.structure_energy,
						training_input.structure_volume
					))

					current_params = training_input.structure_params
					params_strs    = ['%.6E'%g for g in current_params]
					params_strs    = ' '.join(params_strs)
					file.write('Gi  %s\n'%(params_strs))

					atom_idx += 1
				
				progress.update(atom_idx)

			progress.finish()
			file.write('\n')

		if self.log is not None:
			self.log.log("Time Elapsed = %ss"%progress.ttc)
			self.log.unindent()


	# Returns all structures in the training set, divided by which group they
	# are a member of. The result is an array with one element per group and
	# an array with the same format as self.structures for each element. This
	# will also return a second value, which is a unique list of group names.
	#
	# The returned arrays will both be lexicographically sorted by group name.
	def getAllByGroup(self):
		# First, get a unique list of group names.
		group_names = []
		for struct in self.structures:
			group_name = struct[0].group_name
			if group_name not in group_names:
				group_names.append(group_name)

		group_names = sorted(group_names)

		# Now we build the actual array.
		result = []
		for name in group_names:
			current_group = []
			for struct in self.structures:
				if struct[0].group_name == name:
					current_group.append(struct)
			result.append(current_group)

		return result, group_names

	# This function is designed to determine whether the training set can
	# be reasonably split with the given training to validation ratio. If
	# any group in the training set would have no validation data, this 
	# will print a warning and return False. Otherwise, it will return True.
	def generateWarnings(self, validation_ratio):
		if validation_ratio == 1.0:
			return True

		for group, name in zip(*self.getAllByGroup()):
			sub_ratio = validation_ratio - (2 / len(group))

			training_to_select = min([
				int(round(sub_ratio * len(group))),
				len(group) - 2
			])

			if len(group) - 2 - training_to_select == 0:
				msg  = "Structural group \'%s\' is so small that the current "
				msg += "validation ratio would result in it not having any "
				msg += "validation data. Please either change the validation "
				msg += "ratio or turn this warning off."
				msg %= name
				print(msg)
				return False

		return True

	# Extracts all of the relevent cells of information from a line, split
	# on ' ' characters. Also removes '#' characters.
	def _getCellsFromLine(self, line):
		cells = []
		for cell in line.split(" "):
			if cell != '' and not cell.isspace() and cell != '#':
				cells.append(cell)

		return cells

class TrainingInput:
	def fromLines(self, line1, line2, n_lsp):
		cells1 = self._getCellsFromLine(line1)
		self.group_name        = cells1[1]
		self.group_id          = int(cells1[2])
		self.structure_id      = int(cells1[3])
		self.structure_n_atoms = int(cells1[4])
		self.structure_energy  = float(cells1[5])
		self.structure_volume  = float(cells1[6])
		self.structure_params  = np.zeros(n_lsp)

		# All remaining values should be structure parameters.
		for idx, val in enumerate(self._getCellsFromLine(line2)[1:]):
			self.structure_params[idx] = float(val)

		return self


	# Extracts all of the relevent cells of information from a line, split
	# on ' ' characters. Also removes '#' characters.
	def _getCellsFromLine(self, line):
		cells = []
		for cell in line.split(" "):
			if cell != '' and not cell.isspace() and cell != '#':
				cells.append(cell)

		return cells
	