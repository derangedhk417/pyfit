# This file serves to define the valid arguments for pyfit. The information 
# in the structure below is also used to generate help output information when 
# the user specified the --help argument.

import copy
import json

# Returns an object that contains the arguments to the program. Also 
# ensures that all arguments are recognized and that arguments are not
# specified in cases where they are illogical. Will print help information
# if the arguments are incorrect or if the user asks for help. Will also
# print information about a particular argument if the user specifies it
# after --help or -h.
#
# Arguments not specifically enumerated in this file will be returned
# in a list, populating the member 'additional_args' in the returned
# object.
def ParseArgs(arglist):
	program_name = arglist[0]

	# Separate out the config variables that are meant to be passed as arguments.
	cmd_args = {}
	for k in argument_specification:
		if 'long_name' in argument_specification[k]:
			cmd_args[k] = argument_specification[k]

	all_long_names  = [cmd_args[g]['long_name']  for g in cmd_args]
	all_short_names = [cmd_args[g]['short_name'] for g in cmd_args]

	names_to_spec  =     {cmd_args[g]['long_name']: g for g in cmd_args} 
	names_to_spec.update({cmd_args[g]['short_name']: g for g in cmd_args})

	specified_arguments = []

	for arg in arglist[1:]:
		if '=' in arg:
			try:
				argname, argval = arg.split('=')
			except:
				msg  = "All parameters that are not flags should be in the form"
				msg += " --arg-name=arg-value. One of the specified parameters"
				msg += " was in an invalid format." 
				print(msg)
				PrintHelp()
				exit(1)

			specified_arguments.append((argname.lower(), argval))
		else:
			specified_arguments.append((arg.lower(), None))

	# Parse all of the arguments. Also ensure that arguments not specifically
	# enumerated here are in the proper format. Also make sure that there are
	# no duplicate arguments and no illogical argument combinations.
	arg_dictionary  = {}
	additional_args = {}

	for i, arg in enumerate(specified_arguments):
		if arg[0] in ['-h', '--help', 'help']:
			PrintHelp()
			exit(1)

		if arg[0] in all_short_names or arg[0] in all_long_names:
			# We recognize this argument specifically, make sure that
			# there are not any duplicates.
			proper_name = names_to_spec[arg[0]]

			for j, comparison in enumerate(specified_arguments):
				if j != i:
					if comparison[0] in names_to_spec:
						if names_to_spec[comparison[0]] == proper_name:
							# We have a duplicate.
							print("Duplicate argument (%s)"%proper_name)
							PrintHelp()
							exit(1)

			# Now we know that there are no duplicates. Parse the argument.
			typestring = argument_specification[proper_name]['type']
			if typestring == 'flag':
				if arg[1] is not None:
					# This is a flag, but a value was specified.
					print("%s is a flag argument. No value should be specified."%(arg[0]))
					PrintHelp()
					exit(1)

				arg_dictionary[proper_name] = True
			elif typestring == 'string':
				if arg[1] is None:
					print("%s is a value argument. Please specify a value."%(arg[0]))
					PrintHelp()
					exit(1)

				arg_dictionary[proper_name] = arg[1]

			elif typestring == 'int':
				if arg[1] is None:
					print("%s is a value argument. Please specify a value."%(arg[0]))
					PrintHelp()
					exit(1)

				try:
					arg_dictionary[proper_name] = int(arg[1])
				except:
					msg = "%s is an integer argument."%arg[0]
					print(msg)
					PrintHelp()
					exit(1)

		else:
			# This isn't a recognized argument. For now, just assume that its a
			# configuration variable. Make sure there are no duplicates.
			for j, comparison in enumerate(specified_arguments):
				if j != i:
					if arg[0] == comparison[0]:
						# We have a duplicate.
						print("Duplicate argument (%s)"%proper_name)
						PrintHelp()
						exit(1)

			additional_args[arg[0]] = arg[1]


	# We now have a list of arguments that doesn't contain duplicates.
	# Make sure that no argument is specified without its predicate.
	for arg in arg_dictionary:
		predicate = argument_specification[arg]['predicate']
		if predicate is not None:
			# This argument is predicated on the existence of another argument.
			if predicate not in arg_dictionary:
				msg = "The %s argument is only valid if the %s argument is specified."
				msg = msg%(argument_specification[arg]['long_name'], predicate)
				print(msg)
				PrintHelp()
				exit(1)


	# By this point, we have all of the specific arguments validated and
	# parsed. 
	arg_dictionary['additional_args'] = additional_args

	# Add the command line arguments into the config file arguments and rrturn
	# everything as a single object.

	return combineArgsWithConfig(arg_dictionary)


# This function takes the command line arguments and applies them to the 
# configuration file, by replacing anything in the configuration file with
# the values specified in the command line.
# If --config-file (or -j) is not specified, the default is pyfit_config.json.
def combineArgsWithConfig(arg_dict):
	# Figure out where to load the config file from.
	if 'config_file' not in arg_dict:
		config_fpath = 'pyfit_config.json'
	else:
		config_fpath = arg_dict['config_file']

	try:
		# Load it, strip out c-style comments and parse it.
		with open(config_fpath, 'r') as cfile:
			try:
				text   = stripCStyleComments(cfile.read())
				config = json.loads(text)
			except:
				print("Error parsing the config file \'%s\'."%config_fpath)
				exit(1)
	except FileNotFoundError:
		print("The config file '%s' was not found."%config_fpath)
		exit(1)
	except PermissionError:
		print("Access denied for '%s' (configuration file)."%config_fpath)
		exit(1)

	# pyfit requires that a value be specified for everything in the config file,
	# even if you don't happen to be using it on this run. Make sure that there 
	# is at least a key present in the dictionary for everything.
	for k in argument_specification:
		if k not in config:
			# The file doesn't need to reference itself.
			if k != 'config_file':
				msg  = "The configuration variable \'%s\' was missing from the"%k
				msg += " configuration file \'%s\', please add it."%config_fpath
				print(msg)
				exit(1)

	# Add all of the command line arguments into the config structure. 
	for k in arg_dict:
		if k != 'additional_args':
			config[k] = arg_dict[k]

	for k in arg_dict['additional_args']:
		# The additional arguments need to be converted from the --some-name
		# form to the some_name form.
		new_key = k.lstrip('-').replace('-', '_')

		# We need to make an attempt at parsing these values.
		if new_key in argument_specification:
			typestring = argument_specification[new_key]['type']
			if typestring == 'flag':
				if arg_dict['additional_args'][k] is not None:
					msg = "%s is a flag argument. No value should be specified."
					msg %= new_key
					print(msg)
					exit(1)
				config[new_key] = True
			elif typestring == 'string':
				config[new_key] = arg_dict['additional_args'][k]
			elif typestring == 'int':
				try:
					config[new_key] = int(arg_dict['additional_args'][k])
				except:
					msg = "%s is an integer argument."%new_key
					print(msg)
					exit(1)
			elif typestring == 'float':
				try:
					config[new_key] = float(arg_dict['additional_args'][k])
				except:
					msg = "%s is a float argument."%new_key
					print(msg)
					exit(1)
			else:
				config[new_key] = arg_dict['additional_args'][k]

	# Make sure that no unrecognized configuration values are present.
	for k in config:
		if k not in argument_specification:
			msg = "Unrecognized configuration variable \'%s\'"%k
			print(msg)
			exit(1)

	# Make sure that everything matches the expected type.
	for k in config:
		proper_type = argument_specification[k]['type']
		if proper_type == 'string':
			if not isinstance(config[k], str):
				msg  = "Configuration variable \'%s\' must be a string. "%k
				msg += "(%s)"%config_fpath
				print(msg)
				exit(1)
		elif proper_type == 'flag':
			if not isinstance(config[k], bool):
				msg  = "Configuration variable \'%s\' is a flag variable "
				msg += "and must be of type boolean. "
				msg += "(%s)"%config_fpath
				msg %= k
				print(msg)
				exit(1)
		elif proper_type == 'int':
			# Apparently isinstance(True, int) == True in python, hence the
			# extra check in the line below.
			if not isinstance(config[k], int) or isinstance(config[k], bool):
				msg  = "Configuration variable \'%s\' must be an integer. "%k
				msg += "(%s)"%config_fpath
				print(msg)
				exit(1)
		elif proper_type == 'float':
			if not isinstance(config[k], float) or isinstance(config[k], bool):
				msg  = "Configuration variable \'%s\' must be a float. "%k
				msg += "(%s)"%config_fpath
				print(msg)
				exit(1)

	# By this point we are certain that every configuration variable is present,
	# of the correct type and actually recognized.

	# This will construct a runtime type, which makes all of the configuration
	# variables accessible as members. This is more convenient than typing
	# config['var_name'] each time you need to access it.
	return type('config', (), config)

# Strips c-style comments from the string. Assumes that the string is json and
# therefore does not consider single quotes to delimit the start of end of a 
# string of text.
def stripCStyleComments(string):
	lines = string.split('\n')

	result = ''
	for line in lines:
		if not line.strip().startswith('//'):
			# Look for a c-style comment at the end of the line.
			line          = ' ' + line + ' '
			in_quote      = False
			found_comment = False
			for char in range(1, len(line) - 1):
				if line[char - 1] != '\\' and line[char] == '\"':
					in_quote = not in_quote

				if not in_quote:
					if line[char:char + 2] == '//':
						# Everything from this character onwards is a comment.
						result += line[:char]
						found_comment = True
						break

			if not found_comment:
				result += line

	return result


# Prints a listing of arguments, their usage and examples of how to use them.
# If target is specified, will print examples of how to use this particular 
# argument, as well as its long description if available. The target can also
# be a non-argument configuration variable, in which case information from 
# config.py will be printed.
def PrintHelp(target=None):
	all_names = [i for i in argument_specification]


# This structure specifies the type, name, argument name and description of all
# configuration variables that control the functionality of the program. 
# Items with 'short_name' or 'long_name' defined are treated as regular 
# arguments and their usage is printed when the user specifies --help or -h.
# Items without these variables defined ar assumed to be a part of the 
# configuration file. They can only be passed as arguments using their long
# name. For example, if an item is keyed as log_file, it can be specified as
# an argument by putting --log-file=log.txt in the arguments to the program.
# Help information will not be printed for an item like this, unless the user
# types --help config or --help --log-file.
#
# pyfit assumes that a value is given for all arguments in the log file.
# Omission of a configuration variable is not allowed. If one is omitted,
# the program will ask the user if they want it added to the config file for
# them. 
argument_specification = {
	'config_file' : {
		'short_name'       : '-j',
		'long_name'        : '--config-file',
		'type'             : 'string',
		'predicate'        : None,
		'description'      : 'The configuration file to use. Default is pyfit_config.json.',
		'long_description' : None,
		'examples'         : []
	},
	'generate_training_set' : {
		'short_name'       : '-g',
		'long_name'        : '--generate-training-set',
		'type'             : 'flag',
		'predicate'        : None,
		'description'      : 'When specified, the program will generate a training set file',
		'long_description' : None,
		'examples'         : []
	},
	'training_set_output_file' : {
		'short_name'       : '-a',
		'long_name'        : '--training-set-out',
		'type'             : 'string',
		'predicate'        : 'generate_training_set',
		'description'      : 'The file to write the training set to.',
		'long_description' : None,
		'examples'         : []
	},
	'dft_input_directory' : {
		'short_name'       : '-d',
		'long_name'        : '--dft-directory',
		'type'             : 'string',
		'predicate'        : 'generate_training_set',
		'description'      : 'The directory that contains the poscar files.',
		'long_description' : None,
		'examples'         : []
	},
	'dft_input_file' : {
		'short_name'       : '-f',
		'long_name'        : '--dft-file',
		'type'             : 'string',
		'predicate'        : 'generate_training_set',
		'description'      : 'The file that contains the poscar data.',
		'long_description' : None,
		'examples'         : []
	},
	'run_training' : {
		'short_name'       : '-t',
		'long_name'        : '--run-training',
		'type'             : 'flag',
		'predicate'        : None,
		'description'      : 'Train the neural network.',
		'long_description' : None,
		'examples'         : []
	},
	'training_set_in' : {
		'short_name'       : '-s',
		'long_name'        : '--training-set-in',
		'type'             : 'string',
		'predicate'        : 'run_training',
		'description'      : 'The training file to use.',
		'long_description' : None,
		'examples'         : []
	},
	'neural_network_in' : {
		'short_name'       : '-e',
		'long_name'        : '--network-input-file',
		'type'             : 'string',
		'predicate'        : 'run_training',
		'description'      : 'The neural network file to load for training.',
		'long_description' : None,
		'examples'         : []
	},
	'neural_network_out' : {
		'short_name'       : '-y',
		'long_name'        : '--network-output-file',
		'type'             : 'string',
		'predicate'        : 'run_training',
		'description'      : 'The neural network file to write when done training.',
		'long_description' : None,
		'examples'         : []
	},
	'training_iterations' : {
		'short_name'       : '-i',
		'long_name'        : '--training-iterations',
		'type'             : 'int',
		'predicate'        : 'run_training',
		'description'      : 'How many training iterations to run.',
		'long_description' : None,
		'examples'         : []
	},
	'optimizer' : {
		'short_name'       : '-o',
		'long_name'        : '--training-optimizer',
		'type'             : 'string',
		'predicate'        : 'run_training',
		'description'      : 'The optimization algorithm to use. (LBFGS, SGD)',
		'long_description' : None,
		'examples'         : []
	},
	'randomize' : {
		'short_name'       : '-r',
		'long_name'        : '--randomize-nn',
		'type'             : 'flag',
		'predicate'        : 'run_training',
		'description'      : 'Randomize the neural network before training.',
		'long_description' : None,
		'examples'         : []
	},
	'force_cpu' : {
		'short_name'       : '-c',
		'long_name'        : '--force-cpu',
		'type'             : 'flag',
		'predicate'        : 'run_training',
		'description'      : 'Force training to take place on the cpu.',
		'long_description' : None,
		'examples'         : []
	},
	'gpu_affinity' : {
		'short_name'       : '-u',
		'long_name'        : '--gpu-affinity',
		'type'             : 'int',
		'predicate'        : 'run_training',
		'description'      : 'Train the network on the specified gpu (index)',
		'long_description' : None,
		'examples'         : []
	},
	'thread_count' : {
		'short_name'       : '-n',
		'long_name'        : '--n-threads',
		'type'             : 'int',
		'predicate'        : None,
		'description'      : 'Force operations to use only this many threads. This flag does not guarantee that pytorch will not ignore instructions and use more threads anyways. This does guarantee that all operations implemented in pyfit will be limited to this many threads.',
		'long_description' : None,
		'examples'         : []
	},
	'verbosity' : {
		'short_name'       : '-v',
		'long_name'        : '--verbosity',
		'type'             : 'int',
		'predicate'        : None,
		'description'      : 'How verbose the output of the program should be. Default is 1. 0 means don\'t print anything. Values above 4 are treated as 4.',
		'long_description' : None,
	},
	'log_path' : {
		'type'        : 'string',
		'description' : 'The path to the file to put logs into.'
	},
	'neighbor_list_out' : {
		'type'        : 'string',
		'description' : 'The file to write a neighbor list into for each atom in the trainining set.'
	},
	'div_by_r0_squared' : {
		'type'        : 'flag',
		'description' : 'Whether or not to divide structural parameters by their corresponding value of r0 squared.'
	},
	'e_shift' : {
		'type'        : 'float',
		'description' : 'Used to modify energy values placed in the training file while it is being generated. file value = e_DFT + n_atoms*e_shift'
	},
	'network_backup_dir' : {
		'type'        : 'string',
		'description' : 'The directory to place backups of the neural network in during training.'
	},
	'network_backup_interval' : {
		'type'        : 'int',
		'description' : 'The interval to backup the neural network on. 0 = don\'t store backups.'
	},
	'loss_log_path' : {
		'type'        : 'string',
		'description' : 'The file to store the training error of the neural network in. If left blank, error will not be stored.'
	},
	'validation_log_path' : {
		'type'        : 'string',
		'description' : 'The file to store the validation error of the neural network in. If left blank, error will not be stored.'
	},
	'validation_interval' : {
		'type'        : 'int',
		'description' : 'The interval to calculate the validation error on. 0 = don\'t calculate validation.'
	},
	'group_wise_validation_split' : {
		'type'        : 'flag',
		'description' : 'Whether or not to ensure that the validation set is sampled equally for every structural group. This prevents the random selection of validation data from missing too much of one group.'
	},
	'energy_volume_file' : {
		'type'        : 'string',
		'description' : 'The file to store the energy vs. volume data for the network in.'
	},
	'energy_volume_interval' : {
		'type'        : 'int',
		'description' : 'The interval to dump energy vs. volume into the file on.'
	},
	'validation_ratio' : {
		'type'        : 'float',
		'description' : 'The ratio of validation inputs to total inputs.'
	},
	'objective_function' : {
		'type'        : 'string',
		'description' : 'The objective function to minimize when training the network. Options are \'rmse\' and \'group-weights\''
	},
	'unweighted_negative_error' : {
		'type'        : 'flag',
		'description' : 'Controls how the error is calculated when using the \'group-weights\' objective function. If true, groups below their target error will be treated as having zero error. Otherwise, being below target error will cause a group to be treated as having some error.'
	},
	'group_error_file' : {
		'type'        : 'string',
		'description' : 'The file to store the per-group training error in. This option is ignored if objective_function != \'group-weights\''
	},
	'group_error_interval' : {
		'type'        : 'int',
		'description' : 'The interval on which the per-group error is dumped to the group_error_file. This option is ignored if objective_function != \'group-weights\''
	},
	'default_group_target' : {
		'type'        : 'float',
		'description' : 'The default error value to target for groups that don\'t have their target error explicitely given. This option is ignored if objective_function != \'group-weights\''
	},
	'subgroup_error_targets' : {
		'type'        : 'dict',
		'description' : 'The target error value for each group. Groups not included will take the default value of default_group_target. This option is ignored if objective_function != \'group-weights\''
	},
	'learning_rate' : {
		'type'        : 'float',
		'description' : 'The learning rate to use when training the neural network.'
	},
	'max_lbfgs_iterations' : {
		'type'        : 'int',
		'description' : 'The maximum number of LBFGS optimization iterations per training iteration. A value of 10 is usually sufficient.'
	}
}













