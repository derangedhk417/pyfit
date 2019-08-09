# Authors: Adam Robinson, James Hickman
# The file contains code for loading poscar data files. It converts them into
# a structure that is ready for use in neural network training.

import numpy as np
import torch
from util import ProgressBar

# Designed to load a poscar data file or set of poscar data files. Takes
# either a directory or a file as an argument. If a single file is given, 
# will load as many poscar data structures as there are present in the 
# file. If a directory is given, will load all files with the .poscar
# extension that are in the directory.
#
# Once data is loaded, this class can be used as an iterable, returning
# an instance of PoscarStructure for each structure loaded.
class PoscarLoader:
	def __init__(self, e_shift, log=None):
		self.e_shift   = e_shift
		self.loaded    = False
		self.iter      = None
		self.log       = log

		self.n_atoms      = 0
		self.n_structures = 0
		self.structures   = []
		self.all_comments = []

	def loadFromFile(self, file_path):
		if self.log is not None:
			self.log.log("Loading DFT data")
			self.log.indent()

			self.log.log("File = %s"%(file_path))

		with open(file_path, 'r') as file:
			text = file.read()

		text  = text.rstrip()
		lines = text.split("\n")

		progress = ProgressBar("Poscar Files ", 22, len(lines), update_every=50)
		progress.estimate = False
		# This code originally had validation checks for all values.
		# For now, they have been removed. Experience using the program
		# for quite a while has led me to believe that they are uneccessary.

		start_line = 0
		while start_line < len(lines):
			# We need to know the number of atoms in the file 
			# before we can send the proper string of text to
			# the parsing function.
			
			atoms_in_struct = int(lines[start_line + 5])
			
			base   = start_line
			stride = base + 8 + atoms_in_struct
			structure_lines = lines[base:stride]
			
			struct = PoscarStructure(
				structure_lines, 
				self.e_shift
			)
			self.n_atoms += struct.n_atoms
			self.structures.append(struct)

			if struct.comment not in self.all_comments:
				self.all_comments.append(struct.comment)

			start_line += 8 + atoms_in_struct
			progress.update(start_line)

			self.n_structures = len(self.structures)

		progress.finish()
		self.loaded = True

		if self.log is not None:
			self.log.log("Atoms      Loaded = %i"%self.n_atoms)
			self.log.log("Structures Loaded = %i"%self.n_structures)
			self.log.unindent()

		return self

	def __iter__(self):
		if not self.loaded:
			raise Exception("Data has not been loaded.")

		self.iter = self.structures.__iter__()
		return self.iter

	def __next__(self):
		return self.iter.__next__()

	# Returns all unique comment lines that were found in the poscar data.
	# This is often used to store the name of the structural subgroup that
	# the structure corresponds to.
	def getAllComments(self):
		if not self.loaded:
			raise Exception("No data loaded.")

		return self.all_comments

	# Returns all structures whose comment line is an exact match to the 
	# specified comment.
	def getAllByComment(self, comment):
		if not self.loaded:
			raise Exception("No data loaded.")

		result = []
		for s in self.structures:
			if s.comment == comment:
				result.append(s)

		return result

	def loadFromDir(self, dir_path):
		raise Exception("Not Implemented")

# Stores all of the information in a poscar structure.
class PoscarStructure:
	def __init__(self, lines, e_shift, has_force=False):
		self.comment = lines[0]
		self.scale_factor = float(lines[1])
		self.a1           = self._parseVector(lines[2], self.scale_factor)
		self.a2           = self._parseVector(lines[3], self.scale_factor)
		self.a3           = self._parseVector(lines[4], self.scale_factor)
		self.n_atoms      = int(lines[5])

		if lines[6][0] == 'c':
			self.is_cartesian = True
		elif lines[6][0] == 'd':
			self.is_cartesian = False
			raise Exception("Not Implemented.")
		else:
			msg  = "Invalid value encountered in POSCAR structure. "
			msg += "Line 7 should start with 'c' or 'd'"
			raise ValueError(msg)

		self.energy = float(lines[-1]) + (self.n_atoms * e_shift)
		
		self.forces = np.zeros((len(lines[7:-1]), 3))
		self.atoms  = np.zeros((len(lines[7:-1]), 3))
		for idx, line in enumerate(lines[7:-1]):
			cells = self._getCellsFromLine(line)
			self.atoms[idx, :] = [
				float(i) * self.scale_factor for i in cells[:3]
			]

			if len(cells) > 3:
				self.forces[idx, :] = [float(i) for i in cells[3:]]

	def _parseVector(self, string, scale):
		# This function parses a vector supplied as a string of space 
		# separated floating point values. It also scales the vector 
		# based on the supplied scale factor.

		cells = [s for s in string.split(' ') if s != '' and not s.isspace()]

		return np.array([
			float(cells[0])*scale,
			float(cells[1])*scale,
			float(cells[2])*scale
		])

	# Extracts all of the relevent cells of information from a line, split
	# on ' ' characters. Also removes '#' characters.
	def _getCellsFromLine(self, line):
		cells = []
		for cell in line.split(" "):
			if cell != '' and not cell.isspace() and cell != '#':
				cells.append(cell)

		return cells

	def _dumpVector(self, vec):
		return '%.10f %.10f %.10f'%(vec[0], vec[1], vec[2])

	def __str__(self):
		res  = ''
		res += '%s\n'%self.comment
		res += '%f\n'%self.scale_factor
		res += '%s\n'%self._dumpVector(self.a1)
		res += '%s\n'%self._dumpVector(self.a2)
		res += '%s\n'%self._dumpVector(self.a3)
		res += '%i\n'%self.n_atoms
		res += '%s\n'%('c' if self.is_cartesian else 'd')
		for atom, force in zip(self.atoms, self.forces):
			res += '%s %s\n'%(
				self._dumpVector(atom),
				self._dumpVector(force)
			)
		res += '%f\n'%self.energy
		return res
