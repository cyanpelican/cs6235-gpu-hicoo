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
#SBATCH --output=slurm-%j-high-order.out
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
./HiCooExperiment 131072 8 datasets/chicago-crime-comm.tns      NOCPU #for now

./HiCooExperiment 32     8 datasets/enron.tns
./HiCooExperiment 128    8 datasets/enron.tns
./HiCooExperiment 512    8 datasets/enron.tns
./HiCooExperiment 1024   8 datasets/enron.tns

./HiCooExperiment 32     8 datasets/uber.tns
./HiCooExperiment 256    8 datasets/uber.tns
./HiCooExperiment 1024   8 datasets/uber.tns
./HiCooExperiment 4096   8 datasets/uber.tns
./HiCooExperiment 16384  8 datasets/uber.tns
./HiCooExperiment 262114 8 datasets/uber.tns



echo TESTS COMPLETED
