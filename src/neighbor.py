# Authors: Adam Robinson, James Hickman
# This file contains a simple function that takes poscar data and configuration
# information and returns a neighbor list structure. See below for format.

import numpy as np
from util import ProgressBar

class NeighborList:
	def __init__(self, potential, log=None):
		self.config = potential.config
		self.log     = log

		# This will store the indices within the list of
		# atoms that correspond to each structure. This
		# is necessary for calls to .getStructure() to work.
		self.structure_strides = []
		self.atom_neighbors    = []

	def getStructure(self, struct_id):
		# This will return a chunk of neighbor lists corresponding
		# to the structure that was requested.
		start, stride = self.structure_strides[struct_id]

		return self.atom_neighbors[start:stride]

	def getStructureStrides(self):
		return self.structure_strides

	def GenerateNeighborList(self, structures):
		if self.log is not None:
			self.log.log("Generating Neighbor List")
			self.log.indent()

		# For each atom within each structure, we need to generate a list
		# of atoms within the cutoff distance. Periodic images need to be
		# accounted for during this process. Neighbors in this list are
		# specified as coordinates, rather than indices.

		# The final return value of this function in a 3 dimensional list,
		# with the following access structure: 
		#     neighbor = list[structure][atom][neighbor_index]

		# First we will compute the total number of atoms that need to be
		# processed in order to get an estimate of the time this will take
		# to complete.
		n_total = sum([struct.n_atoms**2 for struct in structures])

		progress = ProgressBar("Neighbor List ", 22, n_total, update_every = 25)
		progress.estimate = False
		
		# IMPORTANT NOTE: This needs to be multiplied by 1.5 when PINN 
		#                 gets implemented.
		cutoff = self.config.cutoff_distance * 1.0

		n_processed = 0

		structure_start  = 0
		structure_stride = 0
		for structure in structures:

			# Normalize the translation vectors.
			a1_n = np.linalg.norm(structure.a1)
			a2_n = np.linalg.norm(structure.a2)
			a3_n = np.linalg.norm(structure.a3)

			# Numpy will automatically convert these to arrays when they are 
			# passed to numpy functions, but it will do that each time we call 
			# a function. Converting them beforehand will save some time.
			a1 = structure.a1
			a2 = structure.a2
			a3 = structure.a3


			# Determine the number of times to repeat the
			# crystal structure in each direction.

			x_repeat = int(np.ceil(cutoff / a1_n))
			y_repeat = int(np.ceil(cutoff / a2_n))
			z_repeat = int(np.ceil(cutoff / a3_n))

			# Now we construct an array of atoms that contains all
			# of the repeated atoms that are necessary. We need to 
			# repeat the crystal structure from -repeat*A_n to 
			# positive repeat*A_n. 

			# This is the full periodic structure that we generate.
			# It is a list of vectors, each vector being a length 3
			# list of floating points.
			n_periodic_atoms   = (2*x_repeat + 1)
			n_periodic_atoms  *= (2*y_repeat + 1)
			n_periodic_atoms  *= (2*z_repeat + 1)
			n_periodic_atoms  *= structure.n_atoms
			periodic_structure = np.zeros((n_periodic_atoms, 3))
			atom_idx = 0
			for i in range(-x_repeat, x_repeat + 1):
				for j in range(-y_repeat, y_repeat + 1):
					for k in range(-z_repeat, z_repeat + 1):
						# This is the new location to use as the center
						# of the crystal lattice.
						center_location = a1*i + a2*j + a3*k

						# Now we add each atom + new center location
						# into the periodic structure.
						for atom in structure.atoms:
							val = atom + center_location
							periodic_structure[atom_idx] = val
							atom_idx += 1

			# Here we actually iterate over every atom and then for each atom
			# determine which atoms are within the cutoff distance.
			for atom in structure.atoms:
				# This statement will subtract the current atom position from
				# the position of each potential neighbor, element wise. It 
				# will then calculate the magnitude of each of these vectors 
				# element  wise.
				distances = np.linalg.norm(
						periodic_structure - atom, 
						axis = 1
				)

				# This is special numpy syntax for selecting all items in an 
				# array  that meet a condition. The boolean operators in the 
				# square  brackets actually convert the 'distances' array into 
				# two arrays  of boolean values and then computes their 
				# boolean 'and' operation element wise. It then selects all 
				# items in the array  'periodic_structure' that correspond to 
				# a value of true in the  array of boolean values.
				mask      = (distances > 1e-8) & (distances < cutoff)
				neighbors = periodic_structure[mask]

				# This line just takes all of the neighbor vectors that we now
				# have (as absolute vectors) and changes them into vectors 
				# relative to the atom that we are currently finding neighbors
				# for.
				neighbor_vecs = neighbors - atom

				self.atom_neighbors.append(neighbor_vecs)

				structure_stride += 1

			self.structure_strides.append((structure_start, structure_stride))
			structure_start = structure_stride


			# Update the performance information so we can report
			# progress to the user.
			n_processed += structure.n_atoms**2
			progress.update(n_processed)
		
		progress.update(n_total)
		progress.finish()

		if self.log is not None:
			self.log.log("Time Elapsed = %ss"%progress.ttc)
			self.log.unindent()