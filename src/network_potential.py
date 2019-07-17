# Authors: Adam Robinson, James Hickman
# This file contains the class that represents a neural network potential file,
# including it's weights and biases, as well as the header that describes the
# parameters that it takes.

# this is just used for random number generation
import numpy as np

# this parses and stores the header information
from config import NetworkConfig

class NetworkPotential:
	def __init__(self):
		self.layers         = None
		self.network_values = None

	# Randomizes the network based upon its configuration header value for
	# the maximum random value.
	def randomizeNetwork(self):
		n_values = 0

		previous = self.config.layer_sizes[0]
		for n in self.config.layer_sizes[1:]:
			n_values += n * previous + n
			previous = n

		self.network_values = np.random.uniform(
			-self.config.max_random, 
			self.config.max_random, 
			n_values
		).tolist()
		
		self.config.randomize = False

	def loadFromText(self, text):
		lines = text.rstrip().split('\n')
		cells = [self.getCellsFromLine(line) for line in lines]

		# send the header to the appropriate class for parsing
		self.config = NetworkConfig('\n'.join(lines[:8]))

		if self.config.randomize:
			self.randomizeNetwork()
		
		# If the network needed to be randomized, then the weight and bias
		# values are already loaded into the flat array.
		self._loadNetwork(
			lines, 
			values_loaded=self.config.randomize
		)

	# Writes the network to a file.
	def writeNetwork(self, path):
		if self.layers is None:
			raise Exception("This network has not been loaded.")

		with open(path, 'w') as file:
			file.write(self.config.toFileString())

			# Now we write the weights by column, rather than
			# by row for each layer. The biases go at the end
			# for each layer.
			# len(layer[0][0]) is the width of the weight matrix  (N)
			# len(layer)       is the height of the weight matrix (M)
			for layer in self.layers:
				# Write the weights.
				for weight in range(len(layer[0][0])):
					for node in range(len(layer)):
						weight_value = layer[node][0][weight]
						file.write(' %-+17.8E 0.0000\n'%(weight_value))

				# Write the biases.
				for node in range(len(layer)):
					file.write(' %-+17.8E 0.0000\n'%(layer[node][1]))

	# Using the flat array of values in the network file, constructs an
	# array where each element is a layer. Each element of every layer is
	# a node and each node is an array of length two. The first is an 
	# array of weights in order by the node in the previous layer that
	# they are connected to. The second is the bias of the node.
	#
	# If values_loaded = True, assumes that self.network_values has been
	# initialized already.
	def _loadNetwork(self, lines, values_loaded=False):
		if not values_loaded:
			self.network_values = []
			for line in lines[8:]:
				val = float(self._getCellsFromLine(line)[0])
				self.network_values.append(val)

		self.layers = [[] for i in range(1, self.config.n_layers)]

		# The weights are stored in the network file with the following order:
		#     All of the weights coming from input one are stored first,
		#     in order by which node in the next layer they connect to.
		#     After all of these weights, the biases of the first actual
		#     layer are stored.

		weight_start_offset = 0

		# We need to iterate over each layer and load the corresponding
		# weights for each node in the layer (as well as the biases).
		iterator = zip(self.config.layer_sizes, range(self.config.n_layers))
		for layer_size, idx in iterator:
			# The first "layer" is actually just the inputs, which don't have
			# biases, because this isn't really a layer.

			if idx != 0:
				previous_layer_size = self.config.layer_sizes[idx - 1]
				# We don't carry out this process on the first layer
				# for the above mentioned reason.

				# Here we create an array for each node in this layer.
				# This first index of the array will contain the weights,
				# the second will contain the bias.
				for i in range(layer_size):
					# Skip the first layer, as it is not a real layer.
					self.layers[idx - 1].append([[], 0.0])

				# Weight start offset should have been incremented by the 
				# previous iteration of the loop, and we should be able to 
				# start reading from whatever value it holds.

				# Each weight connected to this node is stored at an offset 
				# equal to the index of the node within the layer.
				# For example, if this is the first node in this layer,
				# the weight from the first node in the previous layer, to
				# this node will be at 
				# self.network_values[weight_start_offset + 0]
				# The weight from the second node in the first layer, to this 
				# node will be at 
				# self.network_values[weight_start_offset + 16 + 0] if this 
				# layer has 16 nodes.
				# If this is the second node in the layer, the above values 
				# would be
				# self.network_values[weight_start_offset + 1] and
				# self.network_values[weight_start_offset + 16 + 1] 
				# respectively.

				for node_index in range(layer_size):
					# For each node in this layer, retrieve the
					# set of weights.
					for weight_index in range(previous_layer_size):
						# self.layers[idx - 1][node_index][0] is the
						# list of weights corresponding to the current node
						# in the current layer.
						offset  = weight_start_offset + weight_index*layer_size
						offset += node_index
						current_weight_value = self.network_values[offset]
						self.layers[idx - 1][node_index][0].append(
							current_weight_value
						)

				# Now that the weights for each node in the layer are read, we
				# need to load the biases.
				bias_offset = previous_layer_size * layer_size
				for node_index in range(layer_size):
					# For each node in this layer, retrieve the bias.
					offset   = weight_start_offset + bias_offset + node_index
					bias_val = self.network_values[offset]
					self.layers[idx - 1][node_index][1] = bias_val

				# We have now loaded all of the relevant information for the 
				# layer. We need to update weight_start_offset so that the next
				# layer starts from the appropriate offset.
				weight_start_offset  = weight_start_offset + bias_offset
				weight_start_offset += layer_size

	# Extracts all of the relevent cells of information from a line, split
	# on ' ' characters. Also removes '#' characters.
	def _getCellsFromLine(self, line):
		cells = []
		for cell in line.split(" "):
			if cell != '' and not cell.isspace() and cell != '#':
				cells.append(cell)

		return cells


			
	