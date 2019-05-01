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
#SBATCH --output=slurm-%j-small-tensors.out
echo $0
echo
ulimit -c unlimited -s
module load cuda

nvidia-smi
echo VISIBLE === $CUDA_VISIBLE_DEVICES

#                 J      BS file
./HiCooExperiment 262114 4 datasets/matmul_2-2-2.tns
./HiCooExperiment 262114 4 datasets/matmul_3-3-3.tns
./HiCooExperiment 262114 4 datasets/matmul_4-3-2.tns
./HiCooExperiment 262114 4 datasets/matmul_4-4-3.tns
./HiCooExperiment 262114 4 datasets/matmul_4-4-4.tns
./HiCooExperiment 262114 4 datasets/matmul_5-5-5.tns
./HiCooExperiment 262114 4 datasets/matmul_6-3-3.tns



./HiCooExperiment 32     8 datasets/vast-2015-mc1-3d.tns
./HiCooExperiment 128    8 datasets/vast-2015-mc1-3d.tns
./HiCooExperiment 256    8 datasets/vast-2015-mc1-3d.tns
./HiCooExperiment 512    8 datasets/vast-2015-mc1-3d.tns

./HiCooExperiment 1024   2 datasets/vast-2015-mc1-3d.tns
./HiCooExperiment 1024   4 datasets/vast-2015-mc1-3d.tns
./HiCooExperiment 1024   8 datasets/vast-2015-mc1-3d.tns
./HiCooExperiment 1024   16 datasets/vast-2015-mc1-3d.tns
./HiCooExperiment 1024   32 datasets/vast-2015-mc1-3d.tns
./HiCooExperiment 1024   64 datasets/vast-2015-mc1-3d.tns
./HiCooExperiment 1024   128 datasets/vast-2015-mc1-3d.tns
./HiCooExperiment 1024   256 datasets/vast-2015-mc1-3d.tns 


echo TESTS COMPLETED
