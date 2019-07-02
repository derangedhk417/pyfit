# This file defines values that can be specified in the configuration file and
# contains logic for parsing the configuration file. Anything that can be 
# specified in the configuration file can also be specified as a command line
# argument.

configuration_spec = {
	'log_path' : {
		'type'        : 'string',
		'default'     : 'pyfit_log.txt',
		'description' : 'The file to write log information to.'
	}
}