# Authors: Adam Robinson, James Hickman
# This file contains classes and methods that make the actual training of
# the neural network work. This includes laying out the structures that
# are necessary for the training process to take place. It also includes 
# the class that stores the neural network in a format prefered by PyTorch
# for running the actual training process. Finally, it includes the actual
# function that trains the neural network, which includes functionality for
# dumping error information during the training.

import numpy        as np
import os
import torch
import torch.nn     as nn
import torch.optim  as optim
import torch.sparse as sparse

from copy import deepcopy
from util import ProgressBar


# Given a training set file and a validation ratio, this class will construct
# a set of torch tensors that are necessary to send input to the neural 
# network and to compute its loss. This includes the following structures:
#     1) The actual energy of each structure.
#     2) An array of the reciprocal number of atoms in each structure.
#     3) An array of structure parameters for direct input into the network.
#     4) The total number of structures being trained on.
#     5) The reduction matrix used to convert atom energies into structure
#        energies.
#
# This class will produce one copy of the above values for training and one
# for validation.
class TorchTrainingData:
	def __init__(self, training_set, validation_ratio, seed=None):
		self.tensor_type = torch.float32
		self.np_type     = np.float32
		self.val_ratio   = validation_ratio
		# The first thing we need to do is split the training set data up into
		# training and validation data. In order for the validation data to be
		# useable, we need to ensure that it represents a proportionate cross
		# section of each group in the training set. We also need to ensure 
		# that the training data contains the largest and the smallest 
		# structure within each group. Since there is significant correlation 
		# between the volume of a structure and the resulting energy, this 
		# needs to be done to ensure that the neural network is interpolating 
		# between the minimum and maximum volume, not extrapolating (which 
		# would be the case if the largest and smallest were in the validation
		# set and not the training set).

		# First, get a list of training inputs by their group.
		by_group, group_names = training_set.getAllByGroup()

		if seed is not None and seed != 0:
			np.random.seed(seed)

		# Within each group, select the largest and smallest structure and put
		# it into the set of structures that will be used for training. Select
		# the appropriate portion of what remains for wach group to split 
		# between validation and training.
		training   = []
		validation = []

		for idx, group in enumerate(by_group):
			# Sort the structures in the group by their volume.
			sorted_group = sorted(
				group, 
				key=lambda x: x[0].structure_volume / x[0].structure_n_atoms
			)

			if validation_ratio == 1.0:
				indices       = np.arange(0, len(sorted_group))
				train_indices = indices
			else:
				# Now we remove the first and last indices and determine what 
				# percentage of what remains needs to be selected for validation.
				training.append(sorted_group[0])
				training.append(sorted_group[-1])

				if len(sorted_group) > 2:
					sorted_group = sorted_group[1:-1]
					proper_ratio = validation_ratio - (2 / len(group))
					indices      = np.arange(0, len(sorted_group))

					# Select this many structures from what remains.
					training_to_select = min([
						int(round(proper_ratio * len(group))),
						len(sorted_group)
					])

					train_indices = np.random.choice(
						indices, 
						training_to_select, 
						replace=False
					)
				else:
					train_indices = []

			val_indices = [i for i in indices if i not in train_indices]

			# We now have a list of structure indices within this group
			# that need to be used to generate training data and a list that
			# need to be used to generate validation data.
			for tidx in train_indices:
				training.append(sorted_group[tidx])

			for vidx in val_indices:
				validation.append(sorted_group[vidx])

		# We now have a list of structures for training and a list of 
		# structures for validation. Send them each to the function that
		# computes all of the necessary tensors for the actual training
		# process.

		# These two tuples are the primary result of this class.
		training_tensors   = self._getTensors(training)

		self.train_energies    = training_tensors[0]
		self.train_reciprocals = training_tensors[1]
		self.train_lsp         = training_tensors[2]
		self.train_n_inputs    = training_tensors[3]
		self.train_reduction   = training_tensors[4]
		self.train_volumes     = training_tensors[5]

		if validation_ratio != 1.0:
			validation_tensors = self._getTensors(validation)

			self.val_energies    = validation_tensors[0]
			self.val_reciprocals = validation_tensors[1]
			self.val_lsp         = validation_tensors[2]
			self.val_n_inputs    = validation_tensors[3]
			self.val_reduction   = validation_tensors[4]
			self.val_volumes     = validation_tensors[5]

	# This function is meant to be called on a set of training structures once
	# the training and validation structures have been separated.
	def _getTensors(self, structures):

		n_atoms           = sum([s[0].structure_n_atoms for s in structures])
		n_params_per_atom = structures[0][0].structure_params.shape[0]
		n_structures      = len(structures)

		energies = [s[0].structure_energy for s in structures]
		energies = torch.tensor([energies], dtype=self.tensor_type)
		energies = energies.transpose(0, 1)

		# This doesn't get used for training, so we can keep it as a python
		# array.
		volumes = []

		for s in structures:
			volumes.append(s[0].structure_volume)

		reciprocals = [1.0 / s[0].structure_n_atoms for s in structures]
		reciprocals = torch.tensor([reciprocals], dtype=self.tensor_type)
		reciprocals = reciprocals.transpose(0, 1)

		lsp = np.zeros((n_atoms, n_params_per_atom), dtype=self.np_type)
		idx = 0
		for struct in structures:
			for atom in struct:
				lsp[idx, :] = atom.structure_params
				idx += 1

		# Using torch.as_tensor instead of torch.tensor will cause the same 
		# underlying buffer to be used, instead of copying it. This is faster.
		lsp = torch.as_tensor(lsp, dtype=self.tensor_type)

		n_inputs = torch.tensor(n_structures, dtype=self.tensor_type)

		# Now we need to perform the most complicated part of this process,
		# constructing the reduction matrix. 
		reduction = np.zeros((n_structures, n_atoms), dtype=self.np_type)
		
		row    = 0
		column = 0
		for struct in structures:
			for atom in struct:
				reduction[row][column] = 1.0
				column += 1
			row += 1

		reduction = torch.as_tensor(reduction, dtype=self.tensor_type)

		return (
			energies,
			reciprocals,
			lsp,
			n_inputs,
			reduction,
			volumes
		)

	def to(self, device):
		self.train_energies    = self.train_energies.to(device)
		self.train_reciprocals = self.train_reciprocals.to(device)
		self.train_lsp         = self.train_lsp.to(device)
		self.train_n_inputs    = self.train_n_inputs.to(device)
		self.train_reduction   = self.train_reduction.to(device)

		if self.val_ratio != 1.0:
			self.val_energies    = self.val_energies.to(device)
			self.val_reciprocals = self.val_reciprocals.to(device)
			self.val_lsp         = self.val_lsp.to(device)
			self.val_n_inputs    = self.val_n_inputs.to(device)
			self.val_reduction   = self.val_reduction.to(device)

		return self

	def cpu(self):
		return self.to('cpu')

# This is the structure that actually gets used for the training process.
class TorchNetwork(nn.Module):
	def __init__(self, network_potential, reduction_matrix, dropout):
		super(TorchNetwork, self).__init__()

		tt = torch.float32

		# Here we need to instantiate and instance of a linear
		# transformation for each real layer of the network and
		# populate its data members with the weights and biases
		# that we want.

		# We have to use the ParameterList class here because the 
		# regular Python list isn't recognized by PyTorch and 
		# subsequent calls to TorchNet.parameters() will not work.
		self.layers           = []
		self.params           = nn.ParameterList()
		self.activation_mode  = network_potential.config.activation_function
		self.reduction_matrix = reduction_matrix
		self.offset           = torch.tensor(0.5)
		self.config           = network_potential.config
		self.dropout          = dropout

		# Create a set of linear transforms.
		for idx in range(len(self.config.layer_sizes)):
			if idx != 0:
				prev_layer_size = self.config.layer_sizes[idx - 1]
				curr_layer_size = self.config.layer_sizes[idx]
				layer           = nn.Linear(prev_layer_size, curr_layer_size)
				current_layer   = network_potential.layers[idx - 1]
				with torch.no_grad():
					t_weight_tensor = [node[0] for node in current_layer]
					t_weight_tensor = torch.tensor(t_weight_tensor, dtype=tt)
					layer.weight.copy_(t_weight_tensor)

					tmp_bias_tensor = [node[1] for node in current_layer]
					tmp_bias_tensor = torch.tensor(tmp_bias_tensor, dtype=tt)
					layer.bias.copy_(tmp_bias_tensor)

				self.layers.append(layer)
				self.params.extend(layer.parameters())

		self.dropout_layer = None
		if self.dropout != 0.0:
			self.dropout_layer = nn.Dropout(p=self.dropout)
			self.forward       = self.dropout_forward

	# This is just used to provide a list of parameters to the
	# optimizer when it is initialized.
	def getParameters(self):
		return self.params

	# This returns the weights and biases for the neural network 
	# in the same format that they were passed in as when initializing
	# the object. 
	def getNetworkValues(self):
		output_layers = []
		layer_copy    = deepcopy(self.layers)

		for layer in layer_copy:
			nodes = []
			for node_idx in range(len(layer.weight.data)):
				node = []
				node.append(layer.weight.cpu().data[node_idx].tolist())
				node.append(layer.bias.cpu().data[node_idx].item())
				nodes.append(node)
			output_layers.append(nodes)

		return output_layers

	# TODO: Probably split this into two functions and dynamically assign 
	#       based on mode. Should be a little faster.
	# TODO: Look into the builtin pytorch method that evaluates a list
	#       of nn.Linear internally. Probably faster.
	# This function actually defines the operation of the Neural Network
	# during feed forward.
	def _inner_forward(self, x):
		# Activation mode 0 is regular sigmoid and mode 
		# 1 is sigmoid shifted by -0.5
		if self.activation_mode == 0:
			x0 = torch.sigmoid(self.layers[0](x))
			for layer in self.layers[1:-1]:
				x0 = torch.sigmoid(layer(x0))
		else:
			x0 = torch.sigmoid(self.layers[0](x)) - self.offset
			for layer in self.layers[1:-1]:
				x0 = torch.sigmoid(layer(x0)) - self.offset

		x0 = self.reduction_matrix.mm(self.layers[-1](x0))
		return x0

	def forward(self, x):
		return self._inner_forward(x)

	def dropout_forward(self, x):
		return self._inner_forward(self.dropout_layer(x))

	def to(self, device):
		super(TorchNetwork, self).to(device)
		self.reduction_matrix = self.reduction_matrix.to(device)
		self.offset           = self.offset.to(device)
		return self

	def cpu(self):
		return self.to('cpu')

	def setReductionMatrix(self, matrix):
		self.reduction_matrix = matrix

	def randomize_self(self):
		for param in self.params:
			if len(param.data.shape) == 2:
				# This is a two dimensional parameter and therefore
				# a set of weights.
				for node_idx in range(len(param.data)):
					for weight_idx in range(len(param.data[node_idx])):
						val = np.random.uniform(-0.5, 0.5)
						param.data[node_idx][weight_idx] = val
						
			elif len(param.data.shape) == 1:
				# This is a one dimensional parameter and therefore
				# an array of biases.
				for node_idx in range(len(param.data)):
					param.data[node_idx] = np.random.uniform(-0.5, 0.5)
					

# This class ties everything together and performs the actual training,
# progress reporting and saving of the resulting files. I considered making
# this a few functions instead of a class, but it occured to me that making
# it into a class would allow me to pretty easily create a stop and resume
# functionality where the entire training process gets saved to disk and 
# loaded from disk in order to resume. I've experienced cases where I 
# underestimated the amount of ram necessary and crashed the training because
# I launched something else midway through. This feature would also be useful
# for people who have a limit on the length of jobs they can run on clusters.
# 
# TODO: Implement pickling and unpickling of this class, as well as resuming
#       training from where the process left off.
class Trainer:
	# The last argument is the config structure generated when the program
	# parses its command line arguments. You can also just as easily make
	# your own if you want to call this code from another program.
	def __init__(self, network_potential, training_set, config, log=None):
		self.log = log

		if self.log is not None:
			self.log.log("Initializing Trainer")
			self.log.indent()

		# Handle enforcement of threading restrictions.
		if config.thread_count != 0:
			os.environ["OMP_NUM_THREADS"] = str(config.thread_count)
			os.environ["MKL_NUM_THREADS"] = str(config.thread_count)
			torch.set_num_threads(config.thread_count)

			if self.log is not None:
				self.log.log("Enforcing --n-threads=%i"%config.thread_count)
				self.log.log("Setting OMP_NUM_THREADS and MKL_NUM_THREADS")
				self.log.log("Calling torch.set_num_threads")

		# In order to actually train a network, we need an optimizer,
		# a loss calculating function, some tensors for that function,
		# a set of inputs and some parameters for how to run the training.
		# The following code sets that up.
		self.seed            = config.validation_split_seed
		self.restart_error   = config.error_restart_level
		self.training_set    = training_set
		self.potential       = network_potential
		self.network_out     = config.neural_network_out
		self.iterations      = config.training_iterations
		self.cpu             = config.force_cpu
		self.gpu             = config.gpu_affinity
		self.threads         = config.thread_count
		self.backup_dir      = config.network_backup_dir
		self.backup_interval = config.network_backup_interval
		self.loss_log        = config.loss_log_path
		self.val_log         = config.validation_log_path
		self.val_interval    = config.validation_interval
		self.energy_file     = config.energy_volume_file
		self.energy_interval = config.energy_volume_interval
		self.learning_rate   = config.learning_rate
		self.max_lbfgs       = config.max_lbfgs_iterations


		self.restarts        = 0
		self.need_to_restart = False

		# Setup the training and validation structures.
		self.dataset = TorchTrainingData(
			training_set,
			config.validation_ratio,
			self.seed
		)

		if self.log is not None:
			if self.seed != 0:
				self.log.log("Used np.random.seed(%i)"%self.seed)
			else:
				self.log.log("Used default seed (probabilistic behavior).")

		# To start, initialize the network with the training set reduction 
		# matrix. This will get switched out temporarily when computing
		# the validation loss.
		self.nn = TorchNetwork(
			network_potential, 
			self.dataset.train_reduction,
			config.dropout
		)

		if torch.cuda.is_available() and not config.force_cpu:
			self.device = torch.device("cuda:%i"%config.gpu_affinity)
		else:
			self.device = torch.device('cpu')

		if self.log is not None:
			self.log.log("Sending Tensors to Device")
			self.log.log("Device = %s"%str(self.device))

		self.nn      = self.nn.to(self.device)
		self.dataset = self.dataset.to(self.device)

		self.optimizer = optim.LBFGS(
			self.nn.getParameters(), 
			lr=self.learning_rate, 
			max_iter=self.max_lbfgs
		)

		if self.log is not None:
			self.log.unindent()

	def loss(self):
		output = self.nn(self.dataset.train_lsp)

		diff        = output - self.dataset.train_energies
		diff_scaled = torch.mul(diff, self.dataset.train_reciprocals)
		sqr_sum     = (diff_scaled**2).sum()
		sqr_sum    /= self.dataset.train_n_inputs
		rmse        = torch.sqrt(sqr_sum)
		return rmse

	def validation_loss(self):
		with torch.no_grad():
			self.nn.setReductionMatrix(self.dataset.val_reduction)
			output = self.nn(self.dataset.val_lsp)

			diff        = output - self.dataset.val_energies
			diff_scaled = torch.mul(diff, self.dataset.val_reciprocals)
			sqr_sum     = (diff_scaled**2).sum()
			sqr_sum    /= self.dataset.val_n_inputs
			rmse        = torch.sqrt(sqr_sum)
			self.nn.setReductionMatrix(self.dataset.train_reduction)
			return rmse.cpu().item()

	def get_structure_energies(self):
		with torch.no_grad():
			output  = self.nn(self.dataset.train_lsp)
			output  = output.cpu().numpy().transpose()[0]
			output *= self.reciprocals
			return output

	def training_closure(self):
		self.optimizer.zero_grad()
		loss = self.loss()

		# Store the loss in the array.
		self.last_loss = loss.cpu().item()

		loss.backward()
		return loss


	# Performs the full training process, as specified in the supplied config
	# structure. This includes writing output files, training the network,
	# etc.
	def train(self):
		if self.log is not None:
			self.log.log("Beginning Training")
			self.log.indent()

		self.training_losses = np.zeros(self.iterations + 1)
		self.iteration       = 0

		if self.val_interval != 0:
			val_size               = (self.iterations // self.val_interval) + 1
			self.validation_losses = np.zeros(val_size)

		if self.energy_interval != 0:
			energy_saves  = (self.iterations // self.energy_interval) + 1
			n_structures  = self.dataset.train_reduction.shape[0]
			self.energies = np.zeros((energy_saves, n_structures))
			self.reciprocals = self.dataset.train_reciprocals
			self.reciprocals = self.reciprocals.cpu().numpy().transpose()[0]

		with torch.no_grad():
			self.last_loss = self.loss().cpu().item()

		if self.log is not None:
			self.log.log("Iterations             = %i"%(self.iterations))
			self.log.log("Validation Interval    = %i"%(self.val_interval))
			self.log.log("Energy Export Interval = %i"%(self.energy_interval))
			self.log.log("Backup Interval        = %i"%(self.backup_interval))
			self.log.log("Backup Location        = %s"%(self.backup_dir))

		# Here we begin the actual training loop.
		try:
			self._train_loop()
		except KeyboardInterrupt as kb:
			# This most likely means the user wants early termination
			# to occur. 
			print("Detected keyboard interrupt, cleaning up . . .")
			if self.log is not None:
				self.log.log("SIGINT detected. Cleaning up . . . ")
				self.log.log("Iteration Reached = %i"%self.iteration)

		if self.need_to_restart:
			self.restart_training_and_randomize()
			return


		# The training is over. Now we write all of the appropriate output 
		# files.

		# Write the loss file.
		with open(self.loss_log, 'w', 1024*10) as file:
			for i in range(self.training_losses.shape[0]):
				file.write('%06i %.10E\n'%(i, self.training_losses[i]))

			if self.log is not None:
				self.log.log("wrote loss log \'%s\'"%(self.loss_log))

		# Write the validation file.
		if self.val_interval != 0:
			with open(self.val_log, 'w', 1024*10) as file:
				for i in range(self.validation_losses.shape[0]):
					line  = '%06i %.10E\n'
					line %= (i * self.val_interval, self.validation_losses[i])
					file.write(line)

			if self.log is not None:
				self.log.log("wrote validation log \'%s\'"%(self.val_log))

		# Write the energy vs. volume file.
		if self.energy_interval != 0:
			with open(self.energy_file, 'w', 1024*10) as file:
				# The first line should be the volume of each structure in
				# order. All subsequent lines should be an iteration and then
				# the energy of each structure in order.
				volumes = self.dataset.train_volumes
				vol_str = ' '.join([str(v) for v in volumes])
				file.write('training_index %s\n'%vol_str)

				for i in range(self.energies.shape[0]):
					energy_str  = ' '.join(['%.6E'%e for e in self.energies[i]])
					line        = '%06i %s\n'
					line       %= (i * self.energy_interval, energy_str)
					file.write(line)

			if self.log is not None:
				self.log.log("wrote energy log \'%s\'"%(self.energy_file))

		# Write the final neural network file.
		layers = self.nn.cpu().getNetworkValues()
		self.potential.layers = layers
		self.potential.writeNetwork(self.network_out)

		if self.log is not None:
			n_out = self.network_out
			self.log.log("wrote final network potential \'%s\'"%(n_out))
			self.log.unindent()

	def restart_training_and_randomize(self):
		self.nn.randomize_self()

		self.optimizer = optim.LBFGS(
			self.nn.getParameters(), 
			lr=self.learning_rate, 
			max_iter=self.max_lbfgs
		)

		self.need_to_restart = False
		self.train()


	# This is the loop that handles the actual training.
	def _train_loop(self):
		progress = ProgressBar(
			"Training ", 
			22, self.iterations + int(self.iterations == 0), 
			update_every = 1
		)

		while self.iteration <= self.iterations:
			progress.update(self.iteration)

			self.training_losses[self.iteration] = self.last_loss

			if self.restart_error != 0.0:
				if self.last_loss > self.restart_error:
					if self.restarts == 3:
						if self.log is not None:
							msg = "Maximum number of restarts exceeded."
							self.log.log(msg)
						break
					else:
						if self.log is not None:
							msg = "Error threshold exceeded, restarting."
							self.log.log(msg)
						self.need_to_restart  = True
						self.restarts        += 1
						break


			# The following lines figure out if we have reached an iteration 
			# where validation information or volume vs. energy information 
			# needs to be stored.
			if self.val_interval != 0:
				if self.iteration % self.val_interval == 0:
					idx  = (self.iteration // self.val_interval)
					self.validation_losses[idx] = self.validation_loss()

			if self.energy_interval != 0:
				if self.iteration % self.energy_interval == 0:
					idx  = (self.iteration // self.energy_interval)
					self.energies[idx, :] = self.get_structure_energies()

			if self.backup_interval != 0:
				if self.iteration % self.backup_interval == 0:
					idx    = self.iteration
					path   = self.backup_dir + 'nn_bk_%05i.nn.dat'%idx
					layers = self.nn.getNetworkValues()
					self.potential.layers = layers
					self.potential.writeNetwork(path)
			
			# Perform an evaluate and correct step, while storing
			# the resulting loss in self.training_losses.
			self.optimizer.step(self.training_closure)

			self.iteration += 1

		progress.finish()
