# pyfit
Flexible system for training neural networks to interpolate DFT data. 

## Notes

James: While you are reviewing this code, if you find a block of code that isn't 100% clear, it may benefit you to insert the following lines under it.

  import code  
  code.interact(local=locals())

This will initialize an interactive console at that line of code and allow you to inspect the contents of all variables that are in scope. I often find this useful when trying to get a sense for how other peoples code works. I also do it when I am trying to figure out how my own code works and I can't remember because its old.

1) You'll notice a lot of error messages split up into multiple lines. This is because I wanted this code to be readable in a terminal, or in an IDE that is split into multiple columns. I didn't enforce a maximum line width in the original version, and it got really annoying.

## Conventions

* Functions meant to be exported from a module have capital letters at the start of each word.
* Classes have the same format.
* module files are camel cased
* Public class methods have a lowercase letter at the beginning of their name.
* Private class methods have an underscore at the beginning of their name.
* Class members are camel cased
* Private class members are camel cased with an underscore prefix.
* All lines should not exceed column 80, unless its extremely awkward to break the line before that.

# TODO

1) Add various checks during the training process for things that could go wrong.
   This should include things like running out of memory, among others.
2) Try to implement a decent means of predicting the time that long running tasks will take.
3) Implement non-cartesian poscar coordinates.
4) Implement a .to method in all classes that store tensors.
5) Decide how to approach logging, without cluttering the code too much.
6) Make sure to log the configuration.
6) Add option to append datetime to certain files.
6) Make it possible to configure pyfit so that it reads gpu affinity from an environment variable.
6) Add seed parameter for if semi-deterministic results are needed. Don't forget torch.seed().
8) Add --slurm option, where the system will automatically pick cpu vs. gpu as well as thread count and affinity based on environment variables.