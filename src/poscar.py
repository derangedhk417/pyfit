# Authors: Adam Robinson, James Hickman
# The file contains code for loading poscar data files. It converts them into
# a structure that is ready for use in neural network training.

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
	def __init__(self, e_shift):
		self.e_shift   = e_shift
		self.loaded    = False
		self.iter      = None

		self.n_atoms      = 0
		self.n_structures = 0
		self.structures   = []
		self.all_comments = []

	def loadFromFile(self, file_path):
		with open(file_path, 'r') as file:
			text = file.read()

		text  = text.rstrip()
		lines = text.split("\n")

		progress = ProgressBar("Poscar Files ", 30, len(lines), update_every=50)

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
			
			struct = PoscarStructure(structure_lines, self.e_shift)
			self.n_atoms += struct.n_atoms
			self.structures.append(struct)

			if struct.comment not in self.all_comments:
				self.all_comments.append(struct.comment)

			start_line += 8 + atoms_in_struct
			progress.update(start_line)

			self.n_structures = len(self.structures)

		progress.finish()
		self.loaded = True

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
	def __init__(self, lines, e_shift):
		self.comment = lines[0]
		self.scale_factor  = float(lines[1])
		self.a1            = self._parseVector(lines[2], self.scale_factor)
		self.a2            = self._parseVector(lines[3], self.scale_factor)
		self.a3            = self._parseVector(lines[4], self.scale_factor)
		self.n_atoms       = int(lines[5])
		

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

		self.atoms = np.zeros((len(lines[7:-1]), 3))
		for idx, line in enumerate(lines[7:-1]):
			self.atoms[idx, :] = self._parseVector(line, self.scale_factor)

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