#!/usr/bin/env bash
# Single-tile OMMA layout check (kernels/cuda_checks/omma_tile.cu) on spark1, vLLM UP (one warp, a few KB):
#   fleetq run --host spark1 --mem 2 --label "nvfp4-mma tile probe" -- scripts/cuda-moe-nvfp4/probe.sh
# Optional SASS comparison of two already-built trees: SASS=1 (needs nv-base and nv-dev from dev1.sh).
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
D=/tmp/nvfp4probe_$FLEETQ_JOB
rsh "mkdir -p $D"
FLEETQ_JOB=$FLEETQ_JOB scp -q "$REPO/kernels/cuda_checks/omma_tile.cu" "$HOST:$D/"
rsh "cd $D && nvcc -O2 --std=c++17 -gencode arch=compute_121a,code=sm_121a omma_tile.cu -o omma_tile && ./omma_tile; cuobjdump -sass omma_tile | grep -m3 OMMA; rm -rf $D"
