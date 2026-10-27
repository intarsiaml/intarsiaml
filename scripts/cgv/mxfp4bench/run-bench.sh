#!/usr/bin/env bash
# Lane cgv: ON spark1 inside a fleetq quiet slot (sources rsynced to ~/intarsiaml-cgv/bench with rsync -aR kernels/abi kernels/common kernels/cuda scripts/cgv/mxfp4bench):
# builds and runs the microbench: shapes x M 1..4, baseline and the multi-row kernel at R = 1 2 4 8 and the production launcher, with 4 rotating weight copies
# (190 MB, the s27d setting) and with 64 (3 GB, what a decode token streams), plus the plain-read DRAM ceiling probe.
cd ~/intarsiaml-cgv/bench/scripts/cgv/mxfp4bench && /usr/local/cuda/bin/nvcc -O3 -arch=sm_121 -I../../../kernels/cuda -o bench bench.cu 2>&1 | grep -i " error"
for nb in 4 64; do echo "== NB=$nb"; NB=$nb ./bench 60; done
for nb in 4 16 64; do echo "== STREAM NB=$nb"; STREAM=1 NB=$nb ./bench 60; done
