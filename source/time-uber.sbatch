#!/bin/bash
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --job-name=gpu-hicoo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --export=ALL
#SBATCH --qos=soc-gpu-kp
#SBATCH --output=slurm-%j-uber.out
echo $0
echo
ulimit -c unlimited -s

nvidia-smi
echo VISIBLE === $CUDA_VISIBLE_DEVICES

# higher-order tensors (crops extra dimensions off)

#                 J      BS file
./HiCooExperiment 32     8 datasets/uber.tns   NOCPU
./HiCooExperiment 256    8 datasets/uber.tns   NOCPU
./HiCooExperiment 1024   8 datasets/uber.tns   NOCPU
./HiCooExperiment 4096   8 datasets/uber.tns   NOCPU
./HiCooExperiment 16384  8 datasets/uber.tns   NOCPU
./HiCooExperiment 262114 8 datasets/uber.tns   NOCPU
./HiCooExperiment 262114 16 datasets/uber.tns  NOCPU
./HiCooExperiment 262114 32 datasets/uber.tns  NOCPU
./HiCooExperiment 262114 64 datasets/uber.tns  NOCPU



echo TESTS COMPLETED
