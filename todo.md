# pyfit todo list

 - Improve matching of help arguments
 - Add various checks during the training process for things that could go wrong. This should include things like running out of memory, among others.
 - Implement non-cartesian poscar coordinates.
 - Use BFG repo cleaner to remove the neural network files from the repo and add them into the zip file.
 - Use PyTorch routines to do the LSP calculations. Even on the CPU they are a lot faster than numpy.
 - Implement my optimized neighbor list algorithm idea, at least as a test on a separate branch.
 - Write documentation for all file formats.
 - Write example code for how to use the classes in this repo to convert your own file formats into the formats used by pyfit.
 - Add the name of each argument to the calls to PrintHelp() in the validateArgs function so that help specific to the argument in question will be printed when there is a misconfiguration.