#!/bin/bash
mv slurm-*.out old-out/
set -e # exit on error


# b. Load CUDA compiler
module load cuda
#unlimit stacksize 


# c. Compile executable
make

# d. Run sbatch script
sbatch script.sbatch
