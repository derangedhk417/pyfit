# This file serves to define the valid arguments for pyfit. The information 
# in the structure below is also used to generate help output information when 
# the user specified the --help argument.

argument_specification = {
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