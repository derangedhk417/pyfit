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
		pass

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

	def loadNetwork(self):
		pass

	def loadFromText(self, text):
		lines = text.rstrip().split('\n')
		cells = [self.getCellsFromLine(line) for line in lines]

		# send the header to the appropriate class for parsing
		self.config = NetworkConfig('\n'.join(lines[:8]))

		if self.config.randomize:
			self.randomizeNetwork()
		else:
			self.loadNetwork()