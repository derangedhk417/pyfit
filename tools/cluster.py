#!/usr/bin/env python3

# Author: Adam Robinson
# This script is meant to make it easier to run pyfit on a cluster from your
# local machine. It is pretty general and will work for other things too
# though. You need sshpass for this to work. This script is meant for slurm
# but it can be modified fairly easily to work for other job managers.
#
# See cluster_template.sh if you want to get this working on your own system.

import numpy as np
import code
import argparse
import os
import time
import sys
import subprocess
import getpass
import json

def run(cmdline):
	process       = subprocess.Popen(cmdline, shell=True)
	output, error = process.communicate()

	if error != '' and error != None:
		raise Exception("Command Error: %s"%error)

def run_cluster(cmdline, cluster, uname, pwd, wd=None):
	if wd is not None:
		run('sshpass -p "%s" ssh -t %s@%s \"cd %s && %s\"'%(
			pwd, uname, cluster, wd, 
			cmdline.replace('"', '\\"').replace('~', '\\~')
		))
	else:
		run('sshpass -p "%s" ssh -t %s@%s \"%s\"'%(
			pwd, uname, cluster, 
			cmdline.replace('"', '\\"').replace('~', '\\~')
		))

def validate_args(args, parser):
	if args.command == []:
		print("No command specified.")
		parser.print_help()
		exit(1)

	# Before we do anything, make sure that the args make sense.
	if args.partition not in ['gpu', 'general', 'debug']:
		print("-p/--partition can only take the values gpu, general and debug.")
		parser.print_help()
		exit(1)

	if args.n_gpu != 0 and args.partition == 'general':
		if not args.no_warn:
			msg  = 'You should probably not use the general queue if you are '
			msg += 'using gpus. Specify --no-warn to run anyways.'
			print(msg)
			parser.print_help()
			exit(1)

	if args.n_gpu < 0:
		print('Negative gpu count specified.')
		parser.print_help()
		exit(1)

	if args.n_gpu != 0:
		if args.partition == 'gpu':
			if args.n_gpu > 24:
				print('Max gpu count for partition \'gpu\' is 24.')
				parser.print_help()
				exit(1)

		if args.partition == 'debug':
			if args.n_gpu > 8:
				print('Max gpu count for partition \'debug\' is 8.')
				parser.print_help()
				exit(1)

	if args.n_cores < 0:
		print('Negative core count specified.')
		parser.print_help()
		exit(1)

	if args.partition == 'gpu':
		if args.n_cores > 960:
			print('Max core count for partition \'gpu\' is 960.')
			parser.print_help()
			exit(1)

	if args.partition == 'debug':
		if args.n_cores > 320:
			print('Max core count for partition \'debug\' is 320.')
			parser.print_help()
			exit(1)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description='Runs a job on the specified cluster, assuming that it ' +
		'uses slurm as its job manager. NOTE: Requires sshpass. ' +
		'(sudo apt install sshpass)',
		epilog='Will create a slurm script, copy it to the cluster and ' +
		'execute it using ssh.'
	)

	parser.add_argument(
		'-c', '--cluster', dest='cluster', type=str, required=True, 
		help='The address of the cluster to run the job on.'
	)

	parser.add_argument(
		'-u', '--username', dest='username', type=str, required=True, 
		help='The username to login with.'
	)

	parser.add_argument(
		'-w', '--working-directory', dest='working_directory', type=str, 
		required=True, help='Absolute path, on cluster to the directory to ' +
		'work in. Will be created if it does not exist.'
	)

	parser.add_argument(
		'-p', '--partition', dest='partition', type=str, required=True, 
		help='The partition to run the job in. (debug, gpu, general)'
	)

	parser.add_argument(
		'-j', '--job-name', dest='job_name', type=str, required=True, 
		help='What to name the job on the cluster.'
	)

	parser.add_argument(
		'-n', '--n-cores', dest='n_cores', type=int, required=True, 
		help='The number of cores to run the job with.'
	)

	parser.add_argument(
		'-t', '--time', dest='time', type=str, default='00:30:00', 
		help='The amount of time that should be allocated for the job. ' +
		'Default is 00:30:00.'
	)

	parser.add_argument(
		'-g', '--n-gpu', dest='n_gpu', type=int, default=0, 
		help='The number of gpus to run the job with. (default 0)'
	)

	parser.add_argument(
		'-y', '--copy', dest='copy', type=str, nargs='*', default=[],
		help='What to copy to the working directory in order to properly ' +
		'run the job. Specify \"./\" to copy the current directory.'
	)

	parser.add_argument(
		'-s', '--slurm-template', dest='slurm_template', type=str, 
		default='%s/cluster_template.sh'%sys.path[0], 
		help='The file to use as a template sbatch script.'
	)

	parser.add_argument(
		dest='command', type=str, nargs='*', default=[],
		help='The command line to run within the slurm script that will be ' +
		'generated and copied to the cluster.'
	)

	args = parser.parse_args()

	validate_args(args, parser)

	if args.working_directory[-1] != '/':
		args.working_directory += '/'

	# Now that the arguments are validated, it's time to ask for the password.

	password = getpass.getpass()

	print("Attempting to start cluster job:")
	print("\tworking directory = %s"%(args.working_directory))
	print("\tpartition         = %s"%(args.partition))
	print("\tgpu count         = %s"%(args.n_gpu))
	print("\tcpu count         = %s"%(args.n_cores))
	print("\ttime              = %s"%(args.time))
	print("\ttemplate          = %s"%(args.slurm_template))
	print("\tcommand           = %s"%(' '.join(args.command)))


	# Now we create the working directory on cluster and copy the necessary 
	# working files over.
	run_cluster(
		'mkdir -p %s'%args.working_directory, 
		args.cluster, args.username, password
	)

	# Copy the working files over.
	if args.copy != []:
		if args.copy == ['./']:
			run('sshpass -p "%s" scp -r ./ %s@%s:%s'%(
				password, args.username, args.cluster, args.working_directory
			))
		else:
			for file in args.copy:
				run('sshpass -p "%s" scp %s %s@%s:%s'%(
					password, file, args.username, 
					args.cluster, args.working_directory
				))

	# Generate an sbatch script and copy it to cluster.

	try:
		with open(args.slurm_template, 'r') as file:
			template = file.read()
	except:
		print("Error reading template file.")
		exit(1)


	script = template.replace('{{{job_name}}}', str(args.job_name))
	script = script.replace('{{{n_gpu}}}',      str(args.n_gpu))
	script = script.replace('{{{n_cores}}}',    str(args.n_cores))
	script = script.replace('{{{partition}}}',  str(args.partition))
	script = script.replace('{{{time}}}',       str(args.time))
	script = script.replace('{{{command}}}', ' '.join(args.command))

	script_name = 'cluster_run_script_tmp.sh'

	# Write the script to a temp file, copy it to cluster and run it.
	try:
		with open(script_name, 'w') as file:
			file.write(script)
	except:
		print("Could not create the script file on local machine.")
		exit(1)


	run('sshpass -p "%s" scp %s %s@%s:%s'%(
		password, script_name, args.username, 
		args.cluster,args.working_directory
	))

	# The file is now on cluster, run it through ssh.

	script_path = args.working_directory + script_name

	run_cluster(
		'chmod +x %s'%script_path, 
		args.cluster, args.username, password
	)

	run_cluster(
		'./%s'%script_name, 
		args.cluster,
		args.username, 
		password, 
		wd=args.working_directory
	)