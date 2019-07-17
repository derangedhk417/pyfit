# Authors: Adam Robinson, James Hickman
# This file contains classes and methods that assist in reading and writing
# training set files in the custom format used for neural network potentials.

# this parses and stores the header information
from config import PotentialConfig
from util   import ProgressBar
class TrainingSet:
	def __init__(self):
		pass

	

	def loadFromFile(self, file_path):
		with open(file_path, 'r') as file:
			text = file.read()

		return self.loadFromText(text)

	def loadFromText(self, text):
		lines = text.rstrip().split('\n')

		self.config = PotentialConfig('\n'.join(lines[:8]))

		self.potential_type = int(self._getCellsFromLine(self.lines[8])[1])
		self.n_structures   = int(self._getCellsFromLine(self.lines[9])[1])
		self.n_atoms        = int(self._getCellsFromLine(self.lines[10])[1])

		
		progress = ProgressBar(
			"Loading Training Set", 
			30, self.n_structures, update_every = 10
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
		while idx < len(self.lines):
			
			atom = TrainingInput(self.lines[idx], self.lines[idx + 1])
			if atom.structure_id != current_id:
				self.structures.append(current_struct)
				current_struct = []
				current_id     = atom.structure_id
				progress.update(current_id + 1)

			current_struct.append(atom)
			idx += 2

		progress.finish()
		self.structures.append(current_struct)

		return self

	# Extracts all of the relevent cells of information from a line, split
	# on ' ' characters. Also removes '#' characters.
	def _getCellsFromLine(self, line):
		cells = []
		for cell in line.split(" "):
			if cell != '' and not cell.isspace() and cell != '#':
				cells.append(cell)

		return cells

class TrainingInput:
	def __init__(self, line1, line2):
		cells1 = self._getCellsFromLine(line1)
		self.group_name        = cells1[1]
		self.group_id          = int(cells1[2])
		self.structure_id      = int(cells1[3])
		self.structure_n_atoms = int(cells1[4])
		self.structure_energy  = float(cells1[5])
		self.structure_volume  = float(cells1[6])
		self.structure_params  = []

		# All remaining values should be structure parameters.
		for i in self._getCellsFromLine(line2)[1:]:
			self.structure_params.append(float(i))

	# Extracts all of the relevent cells of information from a line, split
	# on ' ' characters. Also removes '#' characters.
	def _getCellsFromLine(self, line):
		cells = []
		for cell in line.split(" "):
			if cell != '' and not cell.isspace() and cell != '#':
				cells.append(cell)

		return cells