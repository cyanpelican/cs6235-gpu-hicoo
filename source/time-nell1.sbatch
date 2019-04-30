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
#SBATCH --output=slurm-%j-nell1.out
echo $0
echo
ulimit -c unlimited -s

nvidia-smi
echo VISIBLE === $CUDA_VISIBLE_DEVICES


#                 J      BS file
#./HiCooExperiment 16     8 datasets/nell-1.tns
#./HiCooExperiment 32     8 datasets/nell-1.tns
./HiCooExperiment 16     16 datasets/nell-1.tns
./HiCooExperiment 32     16 datasets/nell-1.tns



echo TESTS COMPLETED
