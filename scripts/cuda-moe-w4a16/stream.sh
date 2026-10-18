#!/usr/bin/env bash
# DRAM read-bandwidth probe by run length (kernels/cuda_checks/stream_runs.cu) on spark1, vLLM UP (1 GB of device memory):
#   fleetq run --host spark1 --mem 4 --label "w4a16 stream probe" -- scripts/cuda-moe-w4a16/stream.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
scp -q "$REPO/kernels/cuda_checks/stream_runs.cu" "$HOST:/tmp/stream_runs_$FLEETQ_JOB.cu"
rsh "cd /tmp && nvcc -O2 -gencode arch=compute_121,code=sm_121 stream_runs_$FLEETQ_JOB.cu -o stream_runs_$FLEETQ_JOB && ./stream_runs_$FLEETQ_JOB && ./stream_runs_$FLEETQ_JOB; rm -f stream_runs_$FLEETQ_JOB stream_runs_$FLEETQ_JOB.cu"
