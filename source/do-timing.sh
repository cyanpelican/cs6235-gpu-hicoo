#!/bin/bash

set -e
./build.sh
./get-datasets.sh

sbatch time-dense.sbatch
sbatch time-flickr.sbatch
sbatch time-high-order-tensors.sbatch
sbatch time-nell1.sbatch
sbatch time-nell2.sbatch
sbatch time-small-tensors.sbatch
