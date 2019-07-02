# This file serves to define the valid arguments for pyfit. The information 
# in the structure below is also used to generate help output information when 
# the user specified the --help argument.

import copy

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
def ParseArgs(self, arglist):
	program_name = arglist[0]

	all_long_names  = [g[i]['long_name']  for g in argument_specification]
	all_short_names = [g[i]['short_name'] for g in argument_specification]

	names_to_spec  =     [{g[i]['long_name']:  g} for g in argument_specification]
	names_to_spec.update([{g[i]['short_name']: g} for g in argument_specification])

	specified_arguments = []

	for arg in arglist[1:]:
		if '=' in arg:
			try:
				argname, argval = arg.split('=')
			except:
				msg  = "All parameters that are not flags should be in the form"
				msg += " --arg-name=arg-value. One of the specified parameters"
				msg += "was in an invalid format." 
				print(msg)
				PrintHelp()
				return None, True
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
			return None, True

		if arg[0] in all_short_names or arg[0] in all_long_names:
			# We recognize this argument specifically, make sure that
			# there are not any duplicates.
			proper_name = names_to_spec[arg[0]]

			for j, comparison in enumerate(specified_arguments):
				if j != i:
					if names_to_spec[comparison[0]] == proper_name:
						# We have a duplicate.
						print("Duplicate argument (%s)"%proper_name)
						PrintHelp()
						return None, True

			# Now we know that there are no duplicates. Parse the argument.
			typestring = argument_specification[proper_name]['type']
			if typestring == 'flag':
				if arg[1] is not None:
					# This is a flag, but a value was specified.
					print("%s is a flag argument. No value should be specified."%(arg[0]))
					PrintHelp()
					return None, True

				arg_dictionary[proper_name] = True
			elif typestring == 'string':
				if arg[1] is None:
					print("%s is a value argument. Please specify a value."%(arg[0]))
					PrintHelp()
					return None, True

				arg_dictionary[proper_name] = arg[1]

			elif typestring == 'int':
				if arg[1] is None:
					print("%s is a value argument. Please specify a value."%(arg[0]))
					PrintHelp()
					return None, True

				try:
					arg_dictionary[proper_name] = int(arg[1])
				except:
					msg = "%s is an integer argument."%arg[0]
					print(msg)
					PrintHelp()
					return None, True

		else:
			# This isn't a recognized argument. For now, just assume that its a
			# configuration variable. Make sure there are no duplicates.
			for j, comparison in enumerate(specified_arguments):
				if j != i:
					if arg[0] == comparison[0]:
						# We have a duplicate.
						print("Duplicate argument (%s)"%proper_name)
						PrintHelp()
						return None, True

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
				return None, True

	# Take everything that wasn'y specified and add a dictionary member for it
	# with its value set to none, or the default value if given.
	for arg in argument_specification:
		if arg not in arg_dictionary:
			current = argument_specification[arg]
			if 'default' in current:
				arg_dictionary[arg] = current['default']
			else:
				if current['type'] == 'flag':
					arg_dictionary[arg] = False
				else:
					arg_dictionary[arg] = None

	# By this point, we have all of the specific arguments validated and
	# parsed. Construct a runtime type and return it.
	arg_dictionary['additional_args'] = additional_args
	arg_dictionary['dict_copy']       = copy.deepcopy(arg_dictionary)

	# There is no guarantee that the configuration parameters specified are 
	# valid, but that should be checked immediately after this.

	return type('arguments', (), arg_dictionary), False


def PrintHelp():
	pass



argument_specification = {
	'config_file' : {
		'short_name'  : '-j',
		'long_name'   : '--config-file',
		'type'        : 'string',
		'default'     : 'pyfit_config.json',
		'predicate'   : None,
		'description' : 'The configuration file to use. Default is pyfit_config.json.',
		'examples'    : []
	},
	'generate_training_set' : {
		'short_name'  : '-g',
		'long_name'   : '--generate-training-set',
		'type'        : 'flag',
		'predicate'   : None,
		'description' : 'When specified, the program will generate a training set file',
		'examples'    : []
	},
	'training_set_output_file' : {
		'short_name'  : '-a',
		'long_name'   : '--training-set-out',
		'type'        : 'string',
		'predicate'   : 'generate_training_set',
		'description' : 'The file to write the training set to.',
		'examples'    : []
	},
	'dft_input_directory' : {
		'short_name'  : '-d',
		'long_name'   : '--dft-directory',
		'type'        : 'string',
		'predicate'   : 'generate_training_set',
		'description' : 'The directory that contains the poscar files.',
		'examples'    : []
	},
	'dft_input_file' : {
		'short_name'  : '-f',
		'long_name'   : '--dft-file',
		'type'        : 'string',
		'predicate'   : 'generate_training_set',
		'description' : 'The file that contains the poscar data.',
		'examples'    : []
	},
	'run_training' : {
		'short_name'  : '-t',
		'long_name'   : '--run-training',
		'type'        : 'flag',
		'predicate'   : None,
		'description' : 'Train the neural network.',
		'examples'    : []
	},
	'training_set_in' : {
		'short_name'  : '-s',
		'long_name'   : '--training-set-in',
		'type'        : 'string',
		'predicate'   : 'run_training',
		'description' : 'The training file to use.',
		'examples'    : []
	},
	'neural_network_in' : {
		'short_name'  : '-e',
		'long_name'   : '--network-input-file',
		'type'        : 'string',
		'predicate'   : 'run_training',
		'description' : 'The neural network file to load for training.',
		'examples'    : []
	},
	'neural_network_out' : {
		'short_name'  : '-y',
		'long_name'   : '--network-output-file',
		'type'        : 'string',
		'predicate'   : 'run_training',
		'description' : 'The neural network file to write when done training.',
		'examples'    : []
	},
	'training_iterations' : {
		'short_name'  : '-i',
		'long_name'   : '--training-iterations',
		'type'        : 'int',
		'predicate'   : 'run_training',
		'description' : 'How many training iterations to run.',
		'examples'    : []
	},
	'optimizer' : {
		'short_name'  : '-o',
		'long_name'   : '--training-optimizer',
		'type'        : 'string',
		'predicate'   : 'run_training',
		'description' : 'The optimization algorithm to use. (LBFGS, SGD)',
		'examples'    : []
	},
	'randomize' : {
		'short_name'  : '-r',
		'long_name'   : '--randomize-nn',
		'type'        : 'flag',
		'predicate'   : 'run_training',
		'description' : 'Randomize the neural network before training.',
		'examples'    : []
	},
	'force_cpu' : {
		'short_name'  : '-c',
		'long_name'   : '--force-cpu',
		'type'        : 'flag',
		'predicate'   : 'run_training',
		'description' : 'Force training to take place on the cpu.',
		'examples'    : []
	},
	'gpu_affinity' : {
		'short_name'  : '-u',
		'long_name'   : '--gpu-affinity',
		'type'        : 'int',
		'predicate'   : 'run_training',
		'description' : 'Train the network on the specified gpu (index)',
		'examples'    : []
	},
	'thread_count' : {
		'short_name'  : '-n',
		'long_name'   : '--n-threads',
		'type'        : 'int',
		'predicate'   : None,
		'description' : 'Force operations to use only this many threads. This flag does not guarantee that pytorch will not ignore instructions and use more threads anyways. This does guarantee that all operations implemented in pyfit will be limited to this many threads.',
		'examples'    : []
	},
	'verbosity' : {
		'short_name'  : '-v',
		'long_name'   : '--verbosity',
		'type'        : 'int',
		'predicate'   : None,
		'description' : 'How verbose the output of the program should be. Default is 1. 0 means don\'t print anything. Values above 4 are treated as 4.'
	}
}