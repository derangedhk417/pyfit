#!/bin/bash

# This script is used by cluster.py to generate a slurm submission script.
# It replaces all of the {{{}}} items with their corresponding command
# line arguments to the cluster.py program. If you carefully modify this,
# and modify parts of cluster.py you should be able to get it working for
# other job managers. You will almost certainly need to change the module
# and source commands to match your system.

rm {{{job_name}}}.sh

current_path=${PWD}

cat > {{{job_name}}}.sh <<!
#!/bin/sh
#SBATCH --gres=gpu:{{{n_gpu}}}
#SBATCH -c {{{n_cores}}}
#SBATCH --partition={{{partition}}}
#SBATCH --time={{{time}}}  
#SBATCH -D $current_path

cd $current_path

module purge
module load powerAI/pytorch-1.5.4
source /opt/DL/pytorch/bin/pytorch-activate
time {{{command}}}

!

sbatch {{{job_name}}}.sh