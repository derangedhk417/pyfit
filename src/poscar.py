# Author: Adam Robinson
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
	def __init__(self, config, e_shift=None):
		self.config    = config
		self.e_shift   = e_shift
		self.loaded    = False
		self.iter      = None

		self.n_atoms      = 0
		self.n_structures = 0
		self.structures   = []
		self.all_comments = []

		if config is None:
			if e_shift is None:
				raise Exception("Either config or e_shift must be specified.")
		else:
			self.e_shift = config.e_shift

	def loadFromFile(self, file_path):
		with open(file_path, 'r') as file:
			text = file.read()

		text  = text.rstrip()
		lines = text.split("\n")

		progress = ProgressBar("Poscar Files ", 30, len(lines), update_every=50)

		start_line = 0
		while start_line < len(lines):
			# We need to know the number of atoms in the file 
			# before we can send the proper string of text to
			# the parsing function.
			try:
				atoms_in_struct = int(lines[start_line + 5])
			except Exception as ex:
				msg  = "Unable to read the number of atoms in the "
				msg += "structure on line %i."%(start_line + 6)
				raise Exception(msg) from ex

			try:
				# Select a chunk of lines corresponding to a structure and send it to 
				# the class that parses structures.
				base   = start_line
				stride = base + 8 + atoms_in_struct
				structure_lines = lines[base:stride]
			except IndexError as ex:
				msg  = "The file appears to be truncated improperly. Attempting "
				msg += "to read the POSCAR structure starting at line %i "
				msg += "resulted in an index error."
				msg %= (start_line + 1)

				raise Exception(msg) from ex

			try:
				struct = PoscarStructure(structure_lines, self.e_shift)
				self.n_atoms += struct.n_atoms
				self.structures.append(struct)

				if struct.comment not in self.all_comments:
					self.all_comments.append(struct.comment)
			except ValueError as ex:
				msg  = "Error occured in POSCAR structure starting on line %i."
				msg %= (start_line + 1)
				raise Exception(msg) from ex


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
		try:
			self.scale_factor  = float(lines[1])
			self.a1            = self.parseVector(lines[2], self.scale_factor)
			self.a2            = self.parseVector(lines[3], self.scale_factor)
			self.a3            = self.parseVector(lines[4], self.scale_factor)
			self.n_atoms       = int(lines[5])
		except ValueError as ex:
			msg = "There was an error parsing a value in a POSCAR structure."
			raise Exception(msg) from ex

		if lines[6][0] == 'c':
			self.is_cartesian = True
		elif lines[6][0] == 'd':
			self.is_cartesian = False
			raise Exception("Not Implemented.")
		else:
			msg  = "Invalid value encountered in POSCAR structure. "
			msg += "Line 7 should start with 'c' or 'd'"
			raise ValueError(msg)


		self.atoms = []
		for i in lines[7:-1]:
			try:
				self.atoms.append(self.parseVector(i, self.scale_factor))
			except ValueError as ex:
				msg  = "Invalid value encountered for atomic coordinate "
				msg += "in POSCAR structure."
				raise ValueError(msg) from ex


		try:
			self.energy = float(lines[-1]) + (self.n_atoms * e_shift)
		except ValueError as ex:
			msg  = "Invalid value encountered for structure energy in "
			msg += "POSCAR structure."
			raise ValueError(msg) from ex

	def parseVector(self, string, scale):
		# This function parses a vector supplied as a string of space separated floating
		# point values. It also scales the vector based on the supplied scale factor.

		cells = [s for s in string.split(' ') if s != '' and not s.isspace()]

		return [
			float(cells[0])*scale,
			float(cells[1])*scale,
			float(cells[2])*scale
		]