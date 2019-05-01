#!/bin/bash

set -e
./build.sh
./get-datasets.sh

sbatch time-dense.sbatch
sbatch time-flickr.sbatch
sbatch time-crime.sbatch #higher-order
sbatch time-enron.sbatch #higher-order
sbatch time-uber.sbatch #higher-order
sbatch time-nell1.sbatch
sbatch time-nell2.sbatch
sbatch time-small-tensors.sbatch
