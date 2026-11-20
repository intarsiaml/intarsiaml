#!/usr/bin/env bash
# storeprefault: compile and run a CUDA microbenchmark on spark2 (run under fleetq --quiet gpu,cpu).
#   CU=d2h3.cu (default); output docs/results/2026-11-20-storeprefault/raw/<name>.txt
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
CU=${CU:-d2h3.cu}; N=${CU%.cu}
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-20-storeprefault/raw; mkdir -p "$OUT"
ssh -n -o BatchMode=yes spark2 "mkdir -p ~/intarsiaml-storeprefault/out"
scp -q -o BatchMode=yes "$REPO/scripts/storeprefault/$CU" "spark2:intarsiaml-storeprefault/out/$CU"
ssh -n -o BatchMode=yes spark2 "cd ~/intarsiaml-storeprefault/out && export PATH=/usr/local/cuda/bin:\$PATH && nvcc -O2 -std=c++17 -o $N $CU && ./$N; nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader" 2>&1 | tee "$OUT/$N.txt"
