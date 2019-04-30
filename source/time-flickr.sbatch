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
#SBATCH --output=slurm-%j-flickr.out
echo $0
echo
ulimit -c unlimited -s

nvidia-smi
echo VISIBLE === $CUDA_VISIBLE_DEVICES

#                 J      BS file
#./HiCooExperiment 16     4 datasets/flickr-3d.tns   NOCPU
#./HiCooExperiment 16     8 datasets/flickr-3d.tns   NOCPU
#./HiCooExperiment 32     8 datasets/flickr-3d.tns   NOCPU
./HiCooExperiment 32     16 datasets/flickr-3d.tns
./HiCooExperiment 64     8 datasets/flickr-3d.tns


echo TESTS COMPLETED
