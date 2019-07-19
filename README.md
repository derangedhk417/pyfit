# pyfit
System for training neural networks to interpolate DFT data. 

## Overview

pyfit uses PyTorch to train feed forward, fully connected neural networks to interpolate DFT potential energies for atomic structures. It is designed specifically to train neural networks using a special feature set developed for this task. This feature set is documented [here](https://www.nature.com/articles/s41467-019-10343-5.pdf?origin=ppub). See the doc/ directory for documentation of the file formats imported and exported by pyfit. 

## What it Does

pyfit performs two primary tasks

1) It converts poscar structures and structure energies into training set files that can be used to train neural network.
2) It trains neural networks using these training set file.

A typical run of pyfit will include the following steps.

1) Load a file that contains DFT data.
2) Load a file that contains neural network weights and biases, as well as parameters that define the feature set for the network.
2) Convert the DFT data into local structure parameters (LSPs). These are the inputs for the neural network.
3) Write a training set file that contains these LSPs and the associated structure energy.
4) Load the neural network weights and biases into either CPU or GPU memory, depending on whats available.
5) Split the training set data up into validation and training data.
6) Train the neural network for the specified number of iterations.

## Getting Started

The easiest way to start using pyfit is to clone this repository, cd into it and run ./install.sh. pyfit gets the majority of its direction from its config file, which is a json file. 

### Config File

This file specifies everything that the program should do when it runs. The default location for it is `src/_pyfit_config.json`. You can override this by passing `--config-file=my_file.json` to the program at the command line. The config file that comes with this repo contains all of the configuration parameters that pyfit supports. You can override them at the command line. pyfit will print help info if you specify `-h` or `--help`. Configuration variables not listed in the help message can still be overriden. Their name is equal to their name in the config json file, with `--` prepended to them and all underscores replaced with hyphens. 

### Running pyfit

#### Generate Training Set

You can run pyfit with no arguments. In this case, it will just use information from the config file to guide the run. In order to get pyfit to convert some DFT data into a training set, you need the following:

1) A file that contains one or more poscar structures with the structure energy attached to them. See doc/ for format.
2) A network potential file. See doc/ for format.

You need to tell pyfit where the DFT data is and where the network potential file is. You also need to specify where you want the output training file to be written. In order to do this, set the following configuration variable:

- `"generate_training_set"    : true`
- `"training_set_output_file" : "some_file"`
- `"dft_input_file"           : "some_file"`
- `"neural_network_in"        : "some_file"`

You will also likely want to set `e_shift`, which can be used to account for a constant offset in all of your DFT data. Each structure that is written to the generated training file will have `e_shift * number of atoms in the structure` added to it.

#### Train a Network

In order to train a neural network, pyfit needs to know what network potential file to start with, what training set file to use and where to write the trained network to when training is done. The following configuration variables are required.

- `"run_training"       : true`
- `"training_set_in"    : "some_file"`
- `"neural_network_in"  : "some_file"`
-	`"neural_network_out" : "some_file"`
