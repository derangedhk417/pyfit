# This file defines values that can be specified in the configuration file and
# contains logic for parsing the configuration file. Anything that can be 
# specified in the configuration file can also be specified as a command line
# argument.

def ArgumentsValid(args):
	pass

def ConstructConfiguration(args):
	pass

# Items in this dictionary that do not have a default value are considered
# required.
configuration_spec = {
	'log_path' : {
		'type'        : 'string',
		'default'     : 'pyfit_log.txt',
		'description' : 'The file to write log information to.'
	},
	'network_input_file' : {
		'type'        : 'string',
		'description' : 'The neural network file to load for training.'
	}
}