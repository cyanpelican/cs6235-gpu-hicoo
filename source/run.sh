#!/bin/bash
mkdir -p old-out/
mv slurm-*.out old-out/
set -e # exit on error


# b. Load CUDA compiler
module load cuda
#unlimit stacksize


# c. Compile executable
./build.sh

# grab datasets
./get-datasets.sh

# d. Run sbatch script
sbatch script.sbatch

while [ ! -f slurm-*.out ]; do sleep .2; done

#until tail -f slurm-*.out | grep -i TESTS\ COMPLETED; do sleep .2; done
tail -f slurm-*.out
