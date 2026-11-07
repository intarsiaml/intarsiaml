#!/usr/bin/env bash
# c27, ON spark1 in a quiet fleetq slot: build and run the WPB microbench (sources under ~/intarsia-c27/src).
cd ~/intarsia-c27/src/scripts/c27/wpb && /usr/local/cuda/bin/nvcc -O3 -arch=sm_121 -I../../../kernels/cuda -o bench bench.cu 2>&1 | grep -i " error"
NB=${NB:-64} ./bench ${REPS:-60}
