#!/usr/bin/env python3

# Author: Adam Robinson
# This script is helpful for producing good quality graphs of the properties 
# of a trained neural network. It only tries to generate the actual graph
# contents. Axis labels and titles are meant to be done elsewhere. I usually
# use keynote to accomplish this.

import code
import argparse
import os
import time
import sys
import torch
import numpy             as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import fnmatch
from   scipy import interpolate

from sys import path
path.append('src')
from potential    import NetworkPotential
from training_set import TrainingSet
from train        import TorchTrainingData, TorchNetwork
from lsp          import computeParameters
from poscar       import PoscarLoader
from util         import ProgressBar


def format_axis(ax):
	ax.xaxis.set_tick_params(width=2)
	ax.yaxis.set_tick_params(width=2)
	
	for tick in ax.xaxis.get_major_ticks():
		tick.label.set_fontsize(20) 
	for tick in ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(20) 
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(1.8)


def get_args():
	parser = argparse.ArgumentParser(
		description='Generates heatmaps of the potential energy surface ' +
		'formed by a neural network potential. In this case, potential ' +
		'energy surface refers to the energy that a probe atom would have ' +
		'at various points in a test structure if all other atoms in the ' +
		'structure are held fixed.',
	)

	parser.add_argument(
		'-n', '--neural-network', dest='network_file', type=str, required=True, 
		help='The neural network to evaluate.'
	)

	parser.add_argument(
		'-d', '--dft-data', dest='dft_file', type=str, required=True, 
		help='The dft structure file to use for renderings.'
	)

	parser.add_argument(
		'-s', '--structure-id', dest='struct_id', type=int, default=-1,
		help='The id of the structure to plot.'
	)

	parser.add_argument(
		'-g', '--group-name', dest='group_name', type=str, default='',
		help='The group name for the structure to plot. If this option is ' +
		'specified, the structure in this group with the lowest energy will ' +
		'be chosen.'
	)

	parser.add_argument(
		'-r', '--resolution', dest='resolution', type=int, default=32,
		help='The square root of the number of points to evaluate in order ' +
		'to render an image. The default is 32, for which a 32x32 square of ' +
		'energy evaluations will be performed to construct a heatmap.'
	)

	parser.add_argument(
		'-w', '--width', dest='width', type=float, default=12.0,
		help='The width, in Angstroms of the square to render.'
	)

	parser.add_argument(
		'-z', '--z-coordinate', dest='z_coordinate', type=float, default=0.0,
		help='The z-coordinate to use for the rendering.'
	)

	parser.add_argument(
		'--draw-atoms', dest='draw_atoms', action='store_true',
		help='Draw a marker for each atom location.'
	)

	parser.add_argument(
		'--radius', dest='radius', type=float, default=0.0,
		help='Draw a circle of this radius around each atom.'
	)

	parser.add_argument(
		'--sweep-dir', dest='sweep_dir', type=str, default='',
		help='The directory to store sweep renderings in.'
	)

	parser.add_argument(
		'--sweep-n', dest='sweep_n', type=int, default=10,
		help='The number of slices to render in the sweep.'
	)

	parser.add_argument(
		'--z-min', dest='z_min', type=float, default=0.0,
		help='Minimum z for sweep rendering.'
	)

	parser.add_argument(
		'--z-max', dest='z_max', type=float, default=0.0,
		help='Maximum z for sweep rendering.'
	)

	parser.add_argument(
		'--e-min', dest='e_min', type=float, default=-5.0,
		help='Minimum energy for sweep rendering.'
	)

	parser.add_argument(
		'--e-max', dest='e_max', type=float, default=-1.0,
		help='Maximum energy for sweep rendering.'
	)

	parser.add_argument(
		'--image-dpi', dest='image_dpi', type=int, default=150,
		help='Final image dpi.'
	)

	parser.add_argument(
		'-c', '--colormap', dest='colormap', type=str, default='RdYlGn',
		help='The colormap to use. If the specified value is a file in the ' +
		'current directory, the file will be loaded and used as a colormap.'
	)

	return parser.parse_args()

def modified_neighbor_list(base, locations, potential):

	n_total = len(locations)

	# In some cases this needs to be multiplied by 1.5.
	# TODO: Figure out exactly when, I haven't encountered this yet.
	cutoff = potential.config.cutoff_distance * 1.0

	n_processed = 0
	# TODO: Implement the optimized neighbor list algorithm I came up with.
	#       It isn't really necessary, but it would be a good test of the
	#       algorithm and could speed the process up by as much as a factor
	#       of ten if I'm right about it. 

	# Normalize the translation vectors.
	a1_n = np.linalg.norm(base.a1)
	a2_n = np.linalg.norm(base.a2)
	a3_n = np.linalg.norm(base.a3)

	# Numpy will automatically convert these to arrays when they are 
	# passed to numpy functions, but it will do that each time we call 
	# a function. Converting them beforehand will save some time.
	a1 = base.a1
	a2 = base.a2
	a3 = base.a3


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
	n_periodic_atoms   = (2*x_repeat + 1)*(2*y_repeat + 1)*(2*z_repeat + 1)
	n_periodic_atoms  *= base.n_atoms
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
				for atom in base.atoms:
					periodic_structure[atom_idx] = atom + center_location
					atom_idx += 1


	# This is kept as a regular python array, as opposed to a numpy array
	# because it is irregular in shape and wouldn't benefit much from being
	# stored as a numpy array.
	structure_neighbor_list = []

	# Here we actually iterate over every atom and then for each atom
	# determine which atoms are within the cutoff distance.
	for atom in locations:
		# This statement will subtract the current atom position from
		# the position of each potential neighbor, element wise. It will
		# then calculate the magnitude of each of these vectors element 
		# wise.
		distances = np.linalg.norm(
			periodic_structure - atom, 
			axis = 1
		)
		# This is special numpy syntax for selecting all items in an array 
		# that meet a condition. The boolean operators in the square 
		# brackets actually convert the 'distances' array into two arrays 
		# of boolean values and then computes their boolean 'and' operation
		# element wise. It then selects all items in the array 
		# 'periodic_structure' that correspond to a value of true in the 
		# array of boolean values.
		mask      = (distances > 1e-8) & (distances < cutoff)
		neighbors = periodic_structure[mask]

		# This line just takes all of the neighbor vectors that we now
		# have (as absolute vectors) and changes them into vectors 
		# relative to the atom that we are currently finding neighbors
		# for.
		neighbor_vecs = neighbors - atom

		structure_neighbor_list.append(neighbor_vecs)

	# Update the performance information so we can report
	# progress to the user.
	n_processed += 1

	return structure_neighbor_list

def render_heatmap(structure, potential, nn, res, width, z, args, save=None):
	# Generate a set of vectors that define all of the locations at which
	# the energy must be evaludated.
	probe_locations = np.zeros((res**2, 3))
	atoms = structure.atoms

	idx = 0
	for xi, x in enumerate(np.linspace(-(width / 2), (width / 2), res)):
		for yi, y in enumerate(np.linspace(-(width / 2), (width / 2), res)):
			probe_locations[idx, :] = [
				x, y, z
			]
			idx += 1
	
	# This is a modified version that treats each location as an atom 
	# that is having its neighbor list computed against the static 
	# structure.
	neighborLists = modified_neighbor_list(
		structure, 
		probe_locations, 
		potential
	)

	# neighborLists now contains a list of neighbors that corresponds to each 
	# probe location, with a proper periodic structure accounted for.

	# Now we compute the structural parameters at each location.

	n_params_per_atom  = potential.config.n_legendre_polynomials
	n_params_per_atom *= potential.config.n_r0
	lsps               = np.zeros((len(neighborLists), n_params_per_atom))

	for i, l in enumerate(neighborLists):
		lsps[i] = computeParameters(l, potential.config)
	
	# Now that the structural parameters have been calculated, we need
	# to evaluate the neural network. For each set.

	if torch.cuda.is_available():
		device = torch.device("cuda:0")
	else:
		device = torch.device('cpu')

	nn = nn.to(device)

	lsps = torch.tensor(lsps, dtype=torch.float32).to(device)
	energies = nn.atomic_forward(lsps).cpu().detach().numpy()

	# Now that we have all of the energies, We need to form a proper grid
	# for plotting.
	grid = energies.T[0].reshape((res, res))

	fig, ax = plt.subplots(1, 1)

	plot = ax.imshow(
		grid, 
		cmap=args.colormap, 
		interpolation='bicubic',
		vmin=args.e_min,
		vmax=args.e_max
	)
	if args.draw_atoms:
		atom_x = np.array([i[0] for i in atoms])
		atom_y = np.array([i[1] for i in atoms])

		atom_x  = ((atom_x - (-(width / 2))) / width) * res
		atom_y  = ((atom_y - (-(width / 2))) / width) * res
		r       = (args.radius / width) * res
	
		if args.radius != 0.0:
			for x, y in zip(atom_x, atom_y):
				c = plt.Circle((x, y), radius=r, ec='red', fc='none')
				ax.add_artist(c)

		sc = ax.scatter(
			atom_x, atom_y, s=50, marker='H', 
			facecolors='none', edgecolors='cyan'
		)
	fig.colorbar(plot)

	if save is not None:
		plt.tight_layout()
		plt.savefig(save, dpi=args.image_dpi)
		plt.close(fig)
	else:
		plt.show()

if __name__ == '__main__':
	args = get_args()

	# Load the training set and neural network.
	potential = NetworkPotential().loadFromFile(args.network_file)
	poscar    = PoscarLoader(0.0).loadFromFile(args.dft_file)

	# Construct a neural network that is ready to be evaluated.
	# Don't specify a reduction matrix, because we aren't training.
	nn = TorchNetwork(potential, None, 0.0)
	
	# Figure out which structure to use.
	structure = None
	if args.struct_id != -1:
		if args.group_name != '':
			print("You must specify either -g or -s, not both.")
			exit(1)

		structure = poscar.structures[args.struct_id]
	else:
		if args.group_name == '':
			print("You must specify either -g or -s.")
			exit(1)

		structures = poscar.getAllByComment(args.group_name)
		structures = sorted(structures, key=lambda x: x.energy)
		structure  = structures[0] # Lowest energy (therefore, likely stable.)


	

	# Keep the third atom in the same plane as the fixed atoms
	# and sweep an 8x8 angrstroem box around them.
	res   = args.resolution
	width = args.width

	if args.sweep_dir != '':
		if not os.path.isdir(args.sweep_dir):
			os.mkdir(args.sweep_dir)

		if args.sweep_dir[-1] != '/':
			args.sweep_dir += '/'

		progress = ProgressBar(
			"Rendering ", 
			22, args.sweep_n, 
			update_every = 1
		)

		sweep = np.linspace(args.z_min, args.z_max, args.sweep_n)
		for idx, z in enumerate(sweep):
			fname = args.sweep_dir + '%05i.png'%idx
			render_heatmap(
				structure, 
				potential,
				nn,
				res,
				width,
				z,
				args,
				save=fname
			)

			progress.update(idx + 1)

		progress.finish()
	else:
		render_heatmap(
			structure, 
			potential,
			nn,
			res,
			width,
			args.z_coordinate,
			args
		)