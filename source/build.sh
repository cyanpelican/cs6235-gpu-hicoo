#!/bin/bash
set -e # exit on error

# b. Load CUDA compiler
module load cuda
#unlimit stacksize 

# c. Compile executable
make
