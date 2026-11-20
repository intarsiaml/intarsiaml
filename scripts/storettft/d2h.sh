#!/usr/bin/env bash
# storettft: compile and run a CUDA microbenchmark on spark2 (run under fleetq --quiet gpu,cpu).
#   CU=d2h.cu (default) or d2h2.cu; output docs/results/2026-11-20-storettft/raw/<name>.txt
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
CU=${CU:-d2h.cu}; N=${CU%.cu}
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
OUT=$REPO/docs/results/2026-11-20-storettft/raw; mkdir -p "$OUT"
scp -q -o BatchMode=yes "$REPO/scripts/storettft/$CU" "spark2:intarsiaml-storettft/out/$CU"
ssh -n -o BatchMode=yes spark2 "cd ~/intarsiaml-storettft/out && export PATH=/usr/local/cuda/bin:\$PATH && nvcc -O2 -o $N $CU && ./$N; nvidia-smi --query-gpu=clocks.sm,power.draw --format=csv,noheader" 2>&1 | tee "$OUT/$N.txt"
