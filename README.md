# pyfit
Flexible system for training neural networks to interpolate DFT data. 

## Notes

1) You'll notice a lot of error messages split up into multiple lines. This is because I wanted this code to be readable in a terminal, or in an IDE that is split into multiple columns. I didn't enforce a maximum line width in the original version, and it got really annoying.

## Conventions

* Functions meant to be exported from a module have capital letters at the start of each wor.
* Classes have the same format.
* module files are camel cased
* Public class methods have a lowercase letter at the beginning of their name.
* Private class methods have an underscore at the beginning of their name.
* Class members are camel cased
* Private class members are camel cased with an underscore prefix.
* All lines should not exceed column 80, unless its extremely awkward to break the line before that.

# TODO

1) Improve help output with examples and drill down into parameters.
2) Consider adding an interactive mode.
3) Should probably add an option that lets the system overwrite existing files.
4) Implement graceful early termination of the training
5) Make sure that the system properly responds to SIGSTOP and SIGCONT when they
   are received during the training process.
6) Add various checks during the training process for things that could go wrong.
   This should include things like running out of memory, among others.
7) Try to implement a decent means of predicting the time that long running tasks will take.
8) Implement non-cartesian poscar coordinates