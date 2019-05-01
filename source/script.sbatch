#!/bin/bash
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --job-name=gpu-hicoo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=60G
#SBATCH --time=00:30:00
#SBATCH --export=ALL
#SBATCH --qos=soc-gpu-kp
ulimit -c unlimited -s

nvidia-smi
echo VISIBLE === $CUDA_VISIBLE_DEVICES

./HiCooExperiment        8
./HiCooExperiment 1024   8 dense-32x32x32
#./HiCooExperiment 16384
#./HiCooExperiment 16384  8 dense-256x256x256d.1 NOCPU
#./HiCooExperiment 262114
#
#./HiCooExperiment 262114 8 datasets/matmul_2-2-2.tns
#./HiCooExperiment 262114 8 datasets/matmul_3-3-3.tns
#./HiCooExperiment 262114 8 datasets/matmul_4-3-2.tns
#./HiCooExperiment 262114 8 datasets/matmul_4-4-3.tns
#./HiCooExperiment 262114 8 datasets/matmul_4-4-4.tns
#./HiCooExperiment 262114 8 datasets/matmul_5-5-5.tns
#./HiCooExperiment 262114 8 datasets/matmul_6-3-3.tns
#
./HiCooExperiment 32     8 datasets/nell-1.tns NOCPU
#./HiCooExperiment 512    8 datasets/nell-2.tns
##./HiCooExperiment 1024   8 datasets/nell-2.tns NOCPU
#./HiCooExperiment 4096   8 datasets/nell-2.tns NOCPU
#./HiCooExperiment 32     8 datasets/flickr-3d.tns NOCPU
#./HiCooExperiment 32     8 datasets/vast-2015-mc1-3d.tns NOCPU

# higher-order tensors (crops extra dimensions off)
#./HiCooExperiment 32     8 datasets/chicago-crime-comm.tns NOCPU
#./HiCooExperiment 32     8 datasets/enron.tns NOCPU
#./HiCooExperiment 32     8 datasets/uber.tns NOPU



echo TESTS COMPLETED
