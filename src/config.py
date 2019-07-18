# Authors: Adam Robinson, James Hickman
# This file contains a class that can both read and write the header that is 
# present in training set files and neural network files.

# WARNING: __eq__ is overridden for this class.
class PotentialConfig:
	def __init__(self):
		pass

	# Extracts all of the relevent cells of information from a line, split
	# on ' ' characters. Also removes '#' characters.
	def _getCellsFromLine(self, line):
		cells = []
		for cell in line.split(" "):
			if cell != '' and not cell.isspace() and cell != '#':
				cells.append(cell)

		return cells


	def loadFromText(self, text):
		lines = text.rstrip().split('\n')
		cells = [self._getCellsFromLine(line) for line in lines]

		# This code originally had validation checks for all values.
		# For now, they have been removed. Experience using the program
		# for quite a while has led me to believe that they are uneccessary.

		self.gi_mode             = int(cells[0][0])
		self.gi_shift            = float(cells[0][1])
		self.activation_function = int(cells[0][2])

		self.n_species = int(cells[1][0])
		self.element   = cells[2][0]
		self.mass      = float(cells[2][1])

		self.randomize = int(cells[3][0])

		# convert from int to bool
		self.randomize = self.randomize == 1

		self.max_random          = float(cells[3][1])
		self.cutoff_distance     = float(cells[3][2])
		self.truncation_distance = float(cells[3][3])
		self.gi_sigma            = float(cells[3][4])

		self.n_legendre_polynomials = int(cells[4][0])
		self.legendre_orders        = [int(c) for c in cells[4][1:]]

		self.n_r0 = int(cells[5][0])
		self.r0   = [float(c) for c in cells[5][1:]]

		self.BOP_param0     = int(cells[6][0])
		self.BOP_parameters = [float(c) for c in cells[6][1:]]

		self.n_layers    = int(cells[7][0])
		self.layer_sizes = [int(c) for c in cells[7][1:]]

		# I'm still including these checks because they've 
		# helped me a few times.
		if len(self.legendre_orders) != self.n_legendre_polynomials:
			error  = "Number of specified legendre polynomials does not match "
			error += "expected value. %i were supposed to be given, but %i "
			error += "were specified."
			error %= (self.n_legendre_polynomials, len(self.legendre_orders)) 
			raise ValueError(error)

		if len(self.r0) != self.n_r0:
			error  = "The number of r0 values declared does not match the "
			error += "actual number present in the file. "
			raise ValueError(error)

		if self.n_layers != len(cells[7][1:]):
			err  = "It appears as though more layers were specified than the "
			err += "first number on line 8 would indicate."
			raise ValueError(err) 

		if self.layer_sizes[0] != self.n_r0 * self.n_legendre_polynomials:
			err  = "The input layer dimensions of the neural network do not "
			err += "match the structural parameter dimensions."
			raise ValueError(err)

		return self

	# This converts the configuration parameters into a string suitable for 
	# writing into a file. If prepend_comment = True, '#' will go at the 
	# beginning of each line. This is useful for LSPARAM files, which have
	# this by convention. If you see something weird and you are wondering
	# why the code does it, the answer is probably something related to 
	# compatibility with any one of the numerous other scripts that exist
	# to interpret this header.
	def toFileString(self, prepend_comment = False):
		string  = ""

		string += ' %i %.7f %i \n'%(
			self.gi_mode, 
			self.gi_shift, 
			self.activation_function
		)

		string += ' %i \n'%self.n_species

		string += ' %s %.7f\n'%(self.element, self.mass)

		string += ' %i %.7f %.7f %.7f %.7f\n'%(
			1 if self.randomize else 0, 
			self.max_random, 
			self.cutoff_distance, 
			self.truncation_distance, 
			self.gi_sigma
		)

		legendre_orders = ' '.join([str(l) for l in self.legendre_orders])
		string += ' %i %s\n'%(self.n_legendre_polynomials, legendre_orders)

		r0_values = ' '.join([str(r) for r in self.r0])
		string += ' %i %s\n'%(self.n_r0, r0_values)

		bop_params = ' '.join([str(b) for b in self.BOP_parameters])
		string += ' %i %s\n'%(self.BOP_param0, bop_params)

		network_layers = ' '.join([str(n) for n in self.layer_sizes])
		string += ' %i %s'%(self.n_layers, network_layers)

		if prepend_comment:
			string = '\n'.join([' #' + line for line in string.split('\n')])

		if string[-1] != '\n':
			string += '\n'

		return string

	# This is used to ensure that the LSPARAM file and the NN file being used
	# match up completely. For those not familiar with the syntax,
	# this allows you to do "config_a == config_b". In Python the default 
	# behavior for class comparison is basically to see if they are the same
	# address, which won't work when comparing the headers from two different
	# files.
	def __eq__(self, other):
		# don't compare the randomize flag
		equality  = self.gi_mode                == other.gi_mode
		equality &= self.gi_shift               == other.gi_shift
		equality &= self.activation_function    == other.activation_function
		equality &= self.n_species              == other.n_species
		equality &= self.element                == other.element
		equality &= self.mass                   == other.mass
		equality &= self.max_random             == other.max_random
		equality &= self.cutoff_distance        == other.cutoff_distance
		equality &= self.truncation_distance    == other.truncation_distance
		equality &= self.gi_sigma               == other.gi_sigma
		equality &= self.n_legendre_polynomials == other.n_legendre_polynomials
		equality &= self.legendre_orders        == other.legendre_orders
		equality &= self.r0                     == other.r0
		equality &= self.BOP_param0             == other.BOP_param0
		equality &= self.BOP_parameters         == other.BOP_parameters
		equality &= self.layer_sizes            == other.layer_sizes

		return equality