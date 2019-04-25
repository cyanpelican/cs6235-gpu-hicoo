#!/bin/bash
#SBATCH --account=soc-gpu-kp
#SBATCH --partition=soc-gpu-kp
#SBATCH --job-name=gpu-hicoo
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:p100:1
#SBATCH --mem=20G
#SBATCH --time=00:1:00
#SBATCH --export=ALL
#SBATCH --qos=soc-gpu-kp     
ulimit -c unlimited -s
#git rev-parse --short HEAD              | tee -a perf.log
#if [[ $(git diff --stat) != '' ]]; then
#  echo '[[dirty]]'                      | tee -a perf.log
#else
#  echo 'clean state'
#fi
#date                                    | tee -a perf.log


./HiCooExperiment "test.dataset"

#echo                                    | tee -a perf.log
