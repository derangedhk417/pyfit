# Authors: Adam Robinson, James Hickman
# This file serves to define the valid arguments for pyfit. The information 
# in the structure below is also used to generate help output information when 
# the user specified the --help argument.

import copy
import json
import util
import os
import sys

from datetime import datetime
from util     import Log

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
				msg  = "All parameters that are not flags should be in the "
				msg += "form --arg-name=arg-value. One of the specified "
				msg += "parameters was in an invalid format." 
				print(msg)
				PrintHelp(arglist)
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
			PrintHelp(arglist)
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
							PrintHelp(arglist)
							exit(1)

			# Now we know that there are no duplicates. Parse the argument.
			typestring = argument_specification[proper_name]['type']
			if typestring == 'flag':
				if arg[1] is not None:
					# This is a flag, but a value was specified.
					msg  = "%s is a flag argument. No value should be "
					msg += "specified."
					msg %= arg[0]
					print(msg)
					PrintHelp(arglist)
					exit(1)

				arg_dictionary[proper_name] = True
			elif typestring == 'string':
				if arg[1] is None:
					msg  = "%s is a value argument. Please specify a value."
					msg %= arg[0]
					print(msg)
					PrintHelp(arglist)
					exit(1)

				arg_dictionary[proper_name] = arg[1]

			elif typestring == 'int':
				if arg[1] is None:
					msg  = "%s is a value argument. Please specify a value."
					msg %= arg[0]
					print(msg)
					PrintHelp(arglist)
					exit(1)

				try:
					arg_dictionary[proper_name] = int(arg[1])
				except:
					msg = "%s is an integer argument."%arg[0]
					print(msg)
					PrintHelp(arglist)
					exit(1)
			elif typestring == 'float':
				if arg[1] is None:
					msg  = "%s is a value argument. Please specify a value."
					msg %= arg[0]
					print(msg)
					PrintHelp(arglist)
					exit(1)

				try:
					arg_dictionary[proper_name] = float(arg[1])
				except:
					msg = "%s is a float argument."%arg[0]
					print(msg)
					PrintHelp(arglist)
					exit(1)

		else:
			# This isn't a recognized argument. For now, just assume that its a
			# configuration variable. Make sure there are no duplicates.
			for j, comparison in enumerate(specified_arguments):
				if j != i:
					if arg[0] == comparison[0]:
						# We have a duplicate.
						print("Duplicate argument (%s)"%proper_name)
						PrintHelp(arglist)
						exit(1)

			additional_args[arg[0]] = arg[1]


	# We now have a list of arguments that doesn't contain duplicates.
	# Make sure that no argument is specified without its predicate.
	for arg in arg_dictionary:
		predicate = argument_specification[arg]['predicate']
		if predicate is not None:
			# This argument is predicated on the existence of another argument.
			if predicate not in arg_dictionary:
				msg  = "The %s argument is only valid if the %s argument is "
				msg += "specified."
				msg %= (argument_specification[arg]['long_name'], predicate)
				print(msg)
				PrintHelp(arglist)
				exit(1)


	# By this point, we have all of the specific arguments validated and
	# parsed. 
	arg_dictionary['additional_args'] = additional_args

	# Add the command line arguments into the config file arguments and return
	# everything as a single object.

	return combineArgsWithConfig(arg_dictionary)


# This function takes the command line arguments and applies them to the 
# configuration file, by replacing anything in the configuration file with
# the values specified in the command line.
# If --config-file (or -j) is 
def combineArgsWithConfig(arg_dict):
	config_default_path = sys.path[0]
	if config_default_path[-1] != '/':
		config_default_path += '/'

	# Figure out where to load the config file from.
	if 'config_file' not in arg_dict:
		config_fpath = config_default_path + '_pyfit_config.json'
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

	# pyfit requires that a value be specified for everything in the config 
	# file, even if you don't happen to be using it on this run. Make sure 
	# that there is at least a key present in the dictionary for everything.
	for k in argument_specification:
		if k not in config:
			# The file doesn't need to reference itself.
			if k != 'config_file':
				msg  = "The configuration variable \'%s\' was missing from the"
				msg += " configuration file \'%s\', please add it."
				msg %= (k, config_fpath)
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
					msg  = "%s is a flag argument. No value should be "
					msg += "specified."
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
		else:
			msg = "Unrecognized configuration variable \'%s\'"%k
			print(msg)
			exit(1)

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

	config['as_dictionary'] = config

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
def PrintHelp(args):
	program_name = 'pyfit.py'
	# Print the whole help message.

	if args is not None:
		printed = False
		for arg in args[1:]:
			if arg != '--help' and arg != '-h':
				printed = True
				DetailedHelp(arg)

		if printed:
			exit(0)

	# We need a list of all short names, all long names and all 
	# descriptions for all arguments.
	short_names  = []
	long_names   = []
	descriptions = []

	for k in argument_specification:
		if 'short_name' in argument_specification[k]:
			short_names.append(argument_specification[k]['short_name'])
			long_names.append(argument_specification[k]['long_name'])
			descriptions.append(argument_specification[k]['description'])

	# Figure out how large the padding needs to be in each column of what 
	# is printed in order to make things look nice.
	short_name_width = max([len(s) for s in short_names])
	long_name_width  = max([len(l) for l in long_names])

	term_width = util.terminal_dims()[1]

	# Now we need to print the short name, long name and description with
	# appropriate padding so everything is aligned and looks nice. We also
	# need to split the descriptions so that they wrap appropriately on 
	# small consoles.

	names_width = 4 + short_name_width + 2 + long_name_width + 4

	help_str  = 'Usage: python3 %s [options]\n\n'%program_name
	help_str += 'python3 pyfit.py -h "variable name" for detailed help.\n\n'
	help_str += 'options:\n'
	for sname, lname, desc in zip(short_names, long_names, descriptions):
		sdiff = short_name_width - len(sname)
		ldiff = long_name_width - len(lname)

		help_str += '    ' + sname + ' '*sdiff + '  ' + lname + ' '*ldiff
		help_str += '    '

		# Now we figure out how the description needs to be printed.
		if len(desc) + names_width < term_width:
			help_str += desc + '\n'
		else:
			# We need to split the description by words and write it
			# with newline characters breaking it up.
			words = desc.split(' ')
			lines = []
			current_word = 0
			current_line = ''
			while current_word < len(words):
				if len(lines) == 0:
					width = len(current_line) + names_width
				else:
					width = len(current_line)


				# See if we can fit another word.
				if width + len(words[current_word]) < term_width:
					line_len = len(current_line)
					if line_len == 0:
						current_line += words[current_word]
					else:
						current_line += ' ' + words[current_word]
				else:
					# Time to create a new line.
					lines.append(current_line)
					current_line = ' '*names_width + words[current_word]

				current_word += 1

			lines.append(current_line)
			help_str += '\n'.join(lines) + '\n'

	print(help_str)

def DetailedHelp(word):
	# Try to convert the keyword into a value from the pyfit
	# arglist and then print the description.
	word = word.lstrip('-').replace('-', ' ')

	# Find the name in the arglist that has the lowest LevenshteinDistance
	# from the word and print it's description.
	full_name = [w for w in argument_specification]
	names     = [w.replace('_', ' ') for w in argument_specification]
	distances = [LevenshteinDistance(w, word) for w in names]
	closest   = full_name[distances.index(min(distances))]

	description = argument_specification[closest]['description']

	print("variable    : %s"%closest)
	print("description : %s"%description)
	print('\n')


# Stolen from https://stackoverflow.com/a/24172422/10470575
def LevenshteinDistance(s1, s2):
    m = len(s1) + 1
    n = len(s2) + 1

    tbl = {}
    for i in range(m):
    	tbl[i, 0] = i
    for j in range(n):
    	tbl[0, j] = j

    for i in range(1, m):
        for j in range(1, n):
            cost  = 0 if s1[i - 1] == s2[j - 1] else 1
            tbl[i, j] = min(
            	tbl[i, j - 1] + 1, 
            	tbl[i - 1, j] + 1, 
            	tbl[i - 1, j - 1] + cost
        	)

    return tbl[i, j]

def ValidateArgs(args):
	# Firstly, make sure that all of the relevant input files exist and are 
	# accessible.

	time_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

	if args.gpu_affinity < 0:
		print("Negative gpu affinity specified.")
		return 1

	if args.thread_count < 0:
		print("Negative thread count specified.")
		return 1

	if args.log_path != "":
		try:
			with open(args.log_path, 'w') as file:
				file.write('test')

			os.remove(args.log_path)
		except:
			print("The log file does not appear to be writable.")
			return 1
	else:
		args.log_path = 'pyfit_log.%s.txt'%time_str

	if args.nvidia_smi_log != '':
		try:
			with open(args.nvidia_smi_log, 'w') as file:
				file.write('test')

			os.remove(args.nvidia_smi_log)
		except:
			print("The nvidia-smi log file does not appear to be writable.")
			return 1

	if args.unique:
		# We need to modify all of the output file names to contain the date
		# & time string.
		def number(name, num):
			parts   = name.split('.')
			result  = parts[0]
			result += '.%05i.%s'%(num, '.'.join(parts[1:]))
			return result
			

		def unique(name):
			if name != "" and not name.isspace():
				idx     = 0
				path_to = '/'.join(name.split('/')[:-1])
				if path_to == '':
					path_to = None

				# This loop looks bad but I sincerely doubt that anyone will
				# every let 2**32 runs go by without cleaning the directory.
				while idx < 2**32:
					all_files = os.listdir(path_to)
					test_name = number(name, idx)
					if test_name not in all_files:
						return test_name
					idx += 1
			else:
				return name

		args.training_set_output_file = unique(args.training_set_output_file)
		args.neural_network_out       = unique(args.neural_network_out)
		args.network_backup_dir       = unique(args.network_backup_dir)
		args.loss_log_path            = unique(args.loss_log_path)
		args.validation_log_path      = unique(args.validation_log_path)
		args.energy_volume_file       = unique(args.energy_volume_file)
		args.log_path                 = unique(args.log_path)
		args.force_train_log          = unique(args.force_train_log)
		args.force_val_log            = unique(args.force_val_log)

		if not os.path.isfile(args.training_set_in) and args.run_training:
			# It's ok if the training set doesn't exist, as long as 
			# we are about to generate it.
			args.training_set_in = args.training_set_output_file


	log = Log(args.log_path)
	
	# ==============================
	# Logging
	# ==============================
	
	log.log("run starting %s"%(time_str))

	# Log the configuration file for this run.
	del args.as_dictionary['as_dictionary']
	arg_dict = args.as_dictionary
	max_len = max([len(k) for k in arg_dict])
	log.log('='*10 + ' Run Configuration ' + 10*'=')
	log.indent()
	for k in arg_dict:
		k_str = k + (max_len - len(k))*' '
		log.log('%s = %s'%(k_str, arg_dict[k]))

	log.unindent()
	log.log('='*39)

	# ==============================
	# End Logging
	# ==============================

	if not args.run_training and not args.generate_training_set:
		print("No task was specified. Use -g and/or -t.\n")
		PrintHelp(None)
		return 1

	if args.neural_network_in == "":
		msg  = "No input neural net was specified. Please specify one via "
		msg += "either the configuration value \'neural_network_in\' or one "
		msg += "of the following arguments:\n"
		msg += "\t--network-input-file=<nn file>\n"
		msg += "\t-e=<nn file>\n\n"
		print(msg)
		PrintHelp(None)
		return 1

	if not os.path.isfile(args.neural_network_in):
		print("The specified input neural network does not exist.")
		return 1

	try:
		with open(args.neural_network_in, 'r') as file:
			file.read(1)
	except:
		print("Could not read the specified neural net input file.")
		return 1

	if args.generate_training_set:
		

		# If the system is generating a training set and neither option is 
		# specified, this doesn't make sense.
		if args.dft_input_file == "":
			msg  = "You have configured the program to generate a training set"
			msg += "but not specified a value for dft_input_file. "
			msg += "Please specify it in the config file or use "
			msg += "the following option: \n"
			msg += "\t--dft-file=<some file>\n\n"
			print(msg)
			PrintHelp(None)
			return 1

		if args.training_set_output_file == "":
			msg  = "You have not specified a value for "
			msg += "training_set_output_file. Please do this in the "
			msg += "configuration file or with one of the  following options:"
			msg += "\n\t--training-set-out=<some file to write>\n"
			msg += "\t-a=<some file to write>\n\n"
			print(msg)
			PrintHelp(None)
			return 1

		if os.path.isfile(args.training_set_output_file) and not args.overwrite:
			msg  = "The specified training set output file already exists. "
			msg += "pyfit does not overwrite large files. Please change the "
			msg += "name or delete it in order to continue."
			print(msg)
			return 1

		# Make sure that nothing is going to stop the system from writing the
		# output file at the end.
		try:
			with open(args.training_set_output_file, 'w') as file:
				file.write('test')

			os.remove(args.training_set_output_file)
		except:
			msg  = "The training set output file does not appear to be "
			msg += "writable."
			print()
			return 1


		if args.dft_input_file != "":
			# Make sure it exists and that we can read it.
			if not os.path.isfile(args.dft_input_file):
				print("The specified dft input file does not exist.")
				return 1

			try:
				with open(args.dft_input_file, 'r') as file:
					file.read(1)
			except:
				print("Could not read the specified dft input file.")
				return 1

	if args.run_training:

		if args.force_interval > 0:
			if args.dft_input_file == "":
				msg  = "You have configured the program to run training with "
				msg += "force optimization but have not specified a value for "
				msg += "dft_input_file. Please specify it in the config file "
				msg += "or use the following option: \n"
				msg += "\t--dft-file=<some file>\n\n"
				print(msg)
				PrintHelp(None)
				return 1

		if args.training_iterations < 0:
			print("Negative number of training iterations specified.")
			return 1

		if args.training_set_in == "":
			msg  = "No input training set was specified. Please specify one "
			msg += "via either the configuration value \'training_set_in\' or "
			msg += "one of the following arguments:\n"
			msg += "\t--training-set-in=<training set file>\n"
			msg += "\t-s=<training set file>\n\n"
			print(msg)
			PrintHelp(None)
			return 1

		if args.neural_network_out == "":
			msg  = "No output neural net was specified. Please specify one  "
			msg += "via either the configuration value \'neural_network_out\' "
			msg += "or one of the following arguments:\n"
			msg += "\t--network-output-file=<nn file>\n"
			msg += "\t-y=<nn file>\n\n"
			print(msg)
			PrintHelp(None)
			return 1

		# Make sure the input files are there and readable. Also make sure that
		# the output file doesn't already exist.

		if not os.path.isfile(args.training_set_in):
			# It's ok if the training set doesn't exist, as long as 
			# we are about to generate it.
			ok  = args.generate_training_set
			ok &= args.training_set_output_file == args.training_set_in

			if not ok:
				print("The specified training set file does not exist.")
				return 1

		try:
			if os.path.isfile(args.training_set_in):
				with open(args.training_set_in, 'r') as file:
					file.read(1)
		except:
			print("Could not read the specified training set input file.")
			return 1

		

		if os.path.isfile(args.neural_network_out) and not args.overwrite:
			print("The specified neural network output file already exists.")
			return 1

		# Make sure that nothing is going to stop the system from writing the
		# output file at the end.
		try:
			with open(args.neural_network_out, 'w') as file:
				file.write('test')

			os.remove(args.neural_network_out)
		except:
			print("The neural net output file does not appear to be writable.")
			return 1

		if args.network_backup_interval > 0:
			if args.network_backup_dir == "":
				msg  = "No network backup directory was specified, but the "
				msg += "backup interval was set. Please either set the "
				msg += "backup interval to 0 or specify a value for "
				msg += "network_backup_dir in the config file."
				print(msg)
				PrintHelp(None)
				return 1

			if args.network_backup_dir[-1] != '/':
				args.network_backup_dir += '/'

			# If the network backup directory already exists, make sure it is 
			# empty. If it doesn't exist, create it. Also make sure that we 
			# can create files in it and write to them.
			if os.path.isdir(args.network_backup_dir):
				try:
					contents = os.listdir(args.network_backup_dir)
				except:
					msg  = "The specified network backup directory is not "
					msg += "accessible. Please check the permissions for it."
					print(msg)
					return 1

				if len(contents) != 0 and not args.overwrite:
					print("The specified backup directory is not empty.")
					return 1
			else:
				try:
					os.mkdir(args.network_backup_dir)
				except:
					print("Could not create the network backup directory.")
					return 1

			try:
				# Make sure that we can write to the backup directory.
				with open(args.network_backup_dir + 'test', 'w') as file:
					file.write('test')

				os.remove(args.network_backup_dir + 'test')
			except:
				msg  = "The network backup directory does not appear "
				msg += "to be writable."
				print(msg)
				return 1
		elif args.network_backup_interval < 0:
			args.network_backup_interval = 0

		if args.validation_interval < 0:
			print("Negative validation interval specified.")
			return 1

		if args.loss_log_path == "":
			msg  = "No path was specified for logging the loss of the neural "
			msg += "network. Please specify a value for \'loss_log_path\' in "
			msg += "the config file."
			print(msg)
			PrintHelp(None)
			return 1

		if args.validation_log_path == "":
			msg  = "No path was specified for logging the validation loss of "
			msg += "the neural network. Please specify a value for "
			msg += "\'loss_log_path\' in the config file."
			print(msg)
			PrintHelp(None)
			return 1

		if args.force_interval < 0:
			msg = "Negative values are invalid for force_interval."
			print(msg)
			PrintHelp(None)
			return 1
		elif args.force_interval > 0:
			# The force training is supposed to take place.
			if args.force_learning_rate < 0.0:
				msg  = "An illogical value was specified for "
				msg += "force_learning_rate."
				print(msg)
				PrintHelp(None)
				return 1

			if args.force_train_log == "":
				msg  = "No path was specified for logging the training loss "
				msg += "of the neural network during force optimization. "
				msg += "Please specify a value for  \'force_train_log\' in "
				msg += "the config file."
				print(msg)
				PrintHelp(None)
				return 1

			if args.force_val_log == "":
				msg  = "No path was specified for logging the validation loss "
				msg += "of the neural network during force optimization. "
				msg += "Please specify a value for  \'force_val_log\' in "
				msg += "the config file."
				print(msg)
				PrintHelp(None)
				return 1

			# Make sure the logs are writeable.
			try:
				with open(args.force_train_log, 'w') as file:
					file.write('test')

				os.remove(args.force_train_log)
			except:
				msg  = "The force optimization training loss log does not "
				msg += "appear to be writeable."
				print(msg)
				return 1

			try:
				with open(args.force_val_log, 'w') as file:
					file.write('test')

				os.remove(args.force_val_log)
			except:
				msg  = "The force optimization validation loss log does not "
				msg += "appear to be writeable."
				print(msg)
				return 1

		if args.energy_volume_file != "":
			try:
				with open(args.energy_volume_file, 'w') as file:
					file.write('test')

				os.remove(args.energy_volume_file)
			except:
				msg  = "An energy vs. volume output file was specified, but "
				msg += "is not writable."
				print(msg)
				return 1

		if args.energy_volume_interval < 0:
			msg  = "Negative value specified for energy vs. volume record "
			msg += "interval."
			print(msg)
			return 1

		if args.energy_volume_interval > 0:
			if args.energy_volume_file == "":
				msg  = "The energy vs. volume record interval is set, but no "
				msg += "file is specified to record it in. Please specify a "
				msg += "value for \'energy_volume_file\' in the config file."
				print(msg)
				return 1

		# Make sure that all of the output files for the training process are
		# writable. 
		try:
			with open(args.loss_log_path, 'w') as file:
				file.write('test')

			os.remove(args.loss_log_path)
		except:
			print("The loss log file does not appear to be writable.")
			return 1

		try:
			with open(args.validation_log_path, 'w') as file:
				file.write('test')

			os.remove(args.validation_log_path)
		except:
			print("The validation log file does not appear to be writable.")
			return 1

		if args.validation_ratio < 0.0 or args.validation_ratio > 1.0:
			print("Validation ratio must be in [0.0, 1.0]")
			return 1

		if args.learning_rate < 0.0 or args.learning_rate > 10:
			print("The learning rate value is illogical, please correct it.")
			return 1

		if args.learning_rate < 0.0 or args.learning_rate > 10:
			print("The learning rate value is illogical, please correct it.")
			return 1

		if args.error_restart_level < 0.0:
			print("The error_restart_level value is illogical.")
			return 1

		if args.validation_ratio == 1.0 and args.validation_interval != 0:
			msg = "No validation data used, but validation interval specified."
			print(msg)
			return 1

		if args.l2_regularization_prefactor < 0.0:
			print("The l2_regularization_prefactor value is illogical.")
			return 1

	return 0, log


# This structure specifies the type, name, argument name and description of all
# configuration variables that control the functionality of the program. 
# Items with 'short_name' or 'long_name' defined are treated as regular 
# arguments and their usage is printed when the user specifies --help or -h.
# Items without these variables defined are assumed to be a part of the 
# configuration file. They can only be passed as arguments using their long
# name. For example, if an item is keyed as log_file, it can be specified as
# an argument by putting --log-file=log.txt in the arguments to the program.
# Help information will not be printed for an item like this, unless the user
# types --help config or --help --log-file.
#
# pyfit assumes that a value is given for all arguments in the config file.
# Omission of a configuration variable is not allowed. If one is omitted,
# the program will tell the user that they need to add it.
config_default_path = sys.path[0]
if config_default_path[-1] != '/':
	config_default_path += '/'

arg_file_path = config_default_path + 'pyfit_arglist.json'
with open(arg_file_path, 'r') as file:
	argument_specification = json.loads(file.read())