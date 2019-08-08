import os
import sys
import copy
import code
import torch
import time
import numpy as np

from args         import ParseArgs, ValidateArgs, PrintHelp
from config       import PotentialConfig
from poscar       import PoscarLoader
from potential    import NetworkPotential
from training_set import TrainingSet
from neighbor     import GenerateNeighborList
from lsp          import GenerateLocalStructureParams
from train        import Trainer
from force        import TorchForceCalculator

if __name__ == '__main__':
	poscar_data = PoscarLoader(0.795023, log=None)
	# poscar_file = '../mini-poscar.dat'
	poscar_file = '../test/input/3-ADAM-DC-L-A-POSCAR-E-full.dat'
	poscar_data = poscar_data.loadFromFile(poscar_file)

	potential   = NetworkPotential(log=None)
	potential   = potential.loadFromFile('../test/nn/best.nn.dat')

	neighborLists = GenerateNeighborList(
		poscar_data.structures,
		potential,
		log=None
	)

	all_atom_neighbors = []
	for struct in neighborLists:
		for atom in struct:
			all_atom_neighbors.append(atom)

	calculator = TorchForceCalculator(torch.float32, potential.config)

	#calculator.loadNeighbors(all_atom_neighbors)

	start = time.time_ns()
	lsp, lspx, lspy, lspz = calculator.generateGradientDisplacementLSPs(
		all_atom_neighbors,
		0.01,
		500	
	)
	time_new = time.time_ns() - start

	start = time.time_ns()
	lsp_old = GenerateLocalStructureParams(
		neighborLists,
		potential.config,
		log=None
	)
	time_old = time.time_ns() - start

	_lsp = []
	for l in lsp_old:
		_lsp.extend(l)

	lsp_old = torch.tensor(_lsp).type(torch.float32)

	res = (lsp - lsp_old).abs()

	code.interact(local=locals())
	