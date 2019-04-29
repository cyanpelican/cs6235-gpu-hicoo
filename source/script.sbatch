#!/bin/bash
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --job-name=gpu-hicoo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=20G
#SBATCH --time=00:10:00
#SBATCH --export=ALL
#SBATCH --qos=soc-gpu-kp
ulimit -c unlimited -s

nvidia-smi

#./HiCooExperiment 
#./HiCooExperiment 1024
#./HiCooExperiment 16384

#./HiCooExperiment 32     datasets/nell-1.tns
./HiCooExperiment 4096  datasets/nell-2.tns


echo TESTS COMPLETED
