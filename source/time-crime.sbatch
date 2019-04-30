#!/bin/bash
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --job-name=gpu-hicoo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --export=ALL
#SBATCH --qos=soc-gpu-kp
#SBATCH --output=slurm-%j-crime.out
echo $0
echo
ulimit -c unlimited -s

nvidia-smi
echo VISIBLE === $CUDA_VISIBLE_DEVICES

# higher-order tensors (crops extra dimensions off)

#                 J      BS file
#./HiCooExperiment 32     8 datasets/chicago-crime-comm.tns  NOCPU
#./HiCooExperiment 256    8 datasets/chicago-crime-comm.tns  NOCPU
#./HiCooExperiment 1024   8 datasets/chicago-crime-comm.tns  NOCPU
#./HiCooExperiment 8192   8 datasets/chicago-crime-comm.tns  NOCPU
./HiCooExperiment 131072 8 datasets/chicago-crime-comm.tns      ###NOCPU #for now



echo TESTS COMPLETED
