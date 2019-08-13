# Author: Adam Robinson
# This file contains code that calculates Local Structure Parameters using
# PyTorch routines, while creating a graph so that the gradient can be 
# calculated. This is done so that the forces on atoms can be calculated 
# and fit to known values when the DFT values for forces are provided.
# This code should also serve as a substitute to the LSP calculation code
# used for generating training sets.

import numpy as np
import torch
import torch.nn.functional as F
import code
from   util import ProgressBar


class TorchLSPCalculator:
	# The config parameter should be an instance of PotentialConfig
	def __init__(self, dtype, config, log=None):
		self.neighbors_loaded = False
		self.dtype            = dtype
		self.config           = config
		self.log              = log

	# Because of the memory required to perform the LSP calculations
	# quickly, this code will divide the neighbor list data into chunks
	# that are small enough to work with and will process them separately
	# when calculating LSPs. The chunks of LSPs produced will be 
	# concatenated at the end. max_chunk is the maximum number of atoms
	# to process at once.
	def generateLSP(self, neighbors, max_chunk=500):
		chunk_start  = 0
		chunk_stride = chunk_start + max_chunk

		lsp = None

		progress = ProgressBar(
			"Structural Parameters ", 
			22, int(len(neighbors) / max_chunk),
			update_every = 5
		)

		idx = 0

		while chunk_start < len(neighbors):
			self.loadNeighbors(neighbors[chunk_start:chunk_stride])
			tmp = self._computeLSP()
			self.cleanupNeighbors()

			if lsp is None:
				lsp = tmp
			else:
				lsp = torch.cat((lsp, tmp), 0)

			chunk_start  += max_chunk
			chunk_stride += max_chunk

			chunk_stride = min(chunk_stride, len(neighbors))

			idx += 1
			progress.update(idx)

		progress.finish()

		return lsp

	# This deletes the rather large arrays of neighbors. The GC should
	# do this automatically, but it never hurts to do it manually.
	def cleanupNeighbors(self):
		del self.neighbor_l
		del self.neighbor_r
		del self.dupl
		self.neighbors_loaded = False

	# This takes a neighbor list structure and converts it into three tensors
	# that are structured so that array operations can be used to compute the
	# LSPs very efficiently. This includes performing LSP calculations for all
	# atoms at once using operations that span all atoms in the training set.
	def loadNeighbors(self, neighbors):
		# Firstly, figure out the highest number of upper diagonal combination
		# of neighbors present in the dataset.
		n_atoms       = len(neighbors)
		max_neighbors = max([
			int((len(n)*len(n) - len(n)) / 2) for n in neighbors
		])
		max_dupl      = max([len(n) for n in neighbors])

		# Now we setup the torch tensor, defaulting all coordinates
		# to 100, 100, 100, which is well outside of the largest 
		# imaginable cutoff radius.

		# We need to have two arrays, since the terms in the LSP calculations
		# are all based on combinations of two neighbors. The third array is 
		# just the array of terms where the combination is of identical atoms
		# instead of two different atoms.
		self.neighbor_l = np.ones((n_atoms, max_neighbors, 3)) * 10000.0
		self.neighbor_r = np.ones((n_atoms, max_neighbors, 3)) * 10000.0
		self.dupl       = np.ones((n_atoms, max_dupl,      3)) * 10000.0

		# Now we iterate through the input neighbor list and create both the
		# left and right arrays for all upper diagonal combinations. We also
		# create the duplicate arrays.
		for i, atom in enumerate(neighbors):
			atom         = np.array(atom)
			length       = atom.shape[0]
			grid         = np.mgrid[0:length, 0:length]
			grid         = grid.swapaxes(0, 2).swapaxes(0, 1)
			m            = grid.shape[0]
			r, c         = np.triu_indices(m, 1)
			combinations = grid[r, c]
			left_array   = atom[combinations[:, 0]]
			right_array  = atom[combinations[:, 1]]
			dupl_array   = atom

			self.neighbor_l[i, :left_array.shape[0]]  = left_array
			self.neighbor_r[i, :right_array.shape[0]] = right_array
			self.dupl[i, :dupl_array.shape[0]]        = dupl_array
				

		# Convert these to the correct type and turn on gradients.
		self.neighbor_l = torch.tensor(self.neighbor_l)
		self.neighbor_r = torch.tensor(self.neighbor_r)
		self.dupl       = torch.tensor(self.dupl)

		self.neighbor_l = self.neighbor_l.type(self.dtype)
		self.neighbor_r = self.neighbor_r.type(self.dtype)
		self.dupl       = self.dupl.type(self.dtype)

		# The neighbor list is constructed as a list of displacement vectors,
		# but we need some kind of absolute coordinates to denote the position
		# of the atom for which each neighbor list is constructed. This is so
		# that PyTorch has something to calculate a gradient with respect to.

		# The class is now setup for force calculations.
		self.neighbors_loaded = True

	# This will calculate the LSPs for the given neighbor list, store them
	# internally and also return them. The shape will be 
	# n_atoms x n_lsp_per_atom. The returned value will be a torch tensor
	# with requires_grad = True
	# 
	# TODO: Make this work for the old gis without the shift.
	def _computeLSP(self):
		if not self.neighbors_loaded:
			msg = "You need to call loadNeighbors before you can call this."
			raise Exception(msg)

		n_atoms = self.neighbor_l.shape[0]

		# First, we need to convert all neighbor vectors into displacement
		# vectors from the atoms. In practice we are basicaly just subtracting
		# zero from each neighbor vector. We need to do this to have a node in
		# the graph for PyTorch to calculate forces with though.
		
		l_disp    = self.neighbor_l
		r_disp    = self.neighbor_r
		dupl_disp = self.dupl


		# Now we need to dot product of the left array with the right array
		# on a per-vector basis. Since this needs to happen across all atoms
		# the expression in the torch.einsum call is extende relative to the
		# numpy.einsum call in the numpy version of this code.
		dot_products      = torch.einsum('ijk,ijk->ij', l_disp, r_disp)
		dot_products_dupl = torch.einsum('ijk,ijk->ij', dupl_disp, dupl_disp)


		# We also need the magnitudes of all vectors.
		left_mag  = torch.norm(l_disp,    dim=2)
		right_mag = torch.norm(r_disp,    dim=2)
		dupl_mag  = torch.norm(dupl_disp, dim=2)


		# Now we can compute the cos(theta_ijk) component for each
		# of the combinations of neighbors.
		angular      = dot_products      / (left_mag * right_mag)
		angular_dupl = dot_products_dupl / (dupl_mag * dupl_mag)


		# Concatenate them together to get the full list.
		angular = torch.cat((angular, angular_dupl), 1)

		s2 = 1.0 / (self.config.gi_sigma ** 2)


		# Now we calculate the cutoff portion of the radial terms.

		d4 = self.config.truncation_distance**4
		#diff            = left_mag - self.config.cutoff_distance
		diff = torch.clamp(
			left_mag - self.config.cutoff_distance, 
			max=0.0
		)
		left_r_minus_rc = diff**4
		left_cutoff     = (left_r_minus_rc / (d4 + left_r_minus_rc))
		#cancel_term     = (0.5 * torch.tanh(-1e6 * (diff)) + 0.5)
		#left_cutoff     = left_cutoff * cancel_term

		#diff             = right_mag - self.config.cutoff_distance
		diff = torch.clamp(
			right_mag - self.config.cutoff_distance, 
			max=0.0
		)
		right_r_minus_rc = diff**4
		right_cutoff     = (right_r_minus_rc / (d4 + right_r_minus_rc))
		#cancel_term      = (0.5 * torch.tanh(-1e6 * (diff)) + 0.5)
		#right_cutoff     = right_cutoff * cancel_term

		#diff            = dupl_mag - self.config.cutoff_distance
		diff = torch.clamp(
			dupl_mag - self.config.cutoff_distance, 
			max=0.0
		)
		dupl_r_minus_rc = diff**4
		dupl_cutoff     = (dupl_r_minus_rc / (d4 + dupl_r_minus_rc))
		#cancel_term     = (0.5 * torch.tanh(-1e6 * (diff)) + 0.5)
		#dupl_cutoff     = dupl_cutoff * cancel_term


		# Turn the r0 values into a tensor.
		r0 = torch.tensor(self.config.r0).type(self.dtype)

		# Perform the appropriate gaussian operation for each r0 value
		# and for each magnitude of each displacement vector combination.

		radial_terms = []
		for r0n in r0:
			left_term = torch.exp(-s2*((left_mag - r0n)**2))
			left_term = left_term * left_cutoff

			right_term = torch.exp(-s2*((right_mag - r0n)**2))
			right_term = right_term * right_cutoff

			dupl_term = torch.exp(-s2*((dupl_mag - r0n)**2))
			dupl_term = dupl_term * dupl_cutoff

			first  = 2 * left_term * right_term
			second = dupl_term**2
			terms  = torch.cat((first, second), 1)
			radial_terms.append(terms)

		

		max_pm           = max(self.config.legendre_orders)
		legendre_results = []

		zeroeth = torch.ones(
			(angular.shape[0], angular.shape[1]), 
			dtype=self.dtype
		)
		legendre_results.append(zeroeth)

		first = angular.clone()
		legendre_results.append(first)

		# Now we calculate the less trivial legendre polynomials using the
		# recursive definition.
		for order in range(1, max_pm):
			current_pm  = (2*order + 1)*angular*legendre_results[order]
			current_pm -= order*legendre_results[order - 1]
			current_pm /= (order + 1)
			legendre_results.append(current_pm)

		# Now we multiply the Legendre Polynomial terms by the radial terms and
		# sum them. This also selects the desired legendre polynomials from the
		# list of those computed. Since the recursive definition is used, legendre
		# polynomials may be computed that aren't actually used in the final 
		# result.

		len_pm = len(self.config.legendre_orders)
		lsps   = []
		for order in self.config.legendre_orders:
			for ridx, r0n in enumerate(radial_terms):
				param = (legendre_results[order] * r0n).sum(dim=1)
				lsps.append(param / (r0[ridx]**2))

		final_lsps       = [l.reshape(n_atoms, 1) for l in lsps]
		final_lsp_tensor = torch.cat((final_lsps[0], final_lsps[1]), 1)

		for i in range(2, len(final_lsps)):
			final_lsp_tensor = torch.cat((final_lsp_tensor, final_lsps[i]), 1)

		if self.config.gi_mode == 1:
			return torch.log(final_lsp_tensor + 0.5)
		elif self.config.gi_mode == 5:
			return torch.log(torch.sqrt(final_lsp_tensor**2 + 1) + final_lsp_tensor)
		else:
			return final_lsp_tensor

	def cpu(self):
		return self.to('cpu')

	def to(self, device):
		if self.neighbors_loaded:
			self.neighbor_l = self.neighbor_l.to(device)
			self.neighbor_r = self.neighbor_r.to(device)
			self.dupl       = self.dupl.to(device)

		return self