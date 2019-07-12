# Author: Adam Robinson
# This file contains a class that can both read and write the header that is 
# present in training set files and neural network files.

class NetworkConfig:
	def __init__(self):
		pass

	# Extracts all of the relevent cells of information from a line, split
	# on ' ' characters. Also removes '#' characters.
	def getCellsFromLine(self, line):
		cells = []
		for cell in line.split(" "):
			if cell != '' and not cell.isspace() and cell != '#':
				cells.append(cell)

		return cells


	def loadFromText(self, text):
		lines = text.rstrip().split('\n')

		cells = [self.getCellsFromLine(line) for line in lines]

		

