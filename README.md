# pyfit
Flexible system for training neural networks to interpolate DFT data. 

# TODO

1) Improve help output with examples and drill down into parameters.
2) Consider adding an interactive mode.
3) Should probably add an option that lets the system overwrite existing files.
4) Implement graceful early termination of the training
5) Make sure that the system properly responds to SIGSTOP and SIGCONT when they
   are received during the training process.
6) Add various checks during the training process for things that could go wrong.
   This should include things like running out of memory, among others.