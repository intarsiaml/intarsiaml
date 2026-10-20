#!/usr/bin/env bash
# Copy result directories of ~/intarsia-decfuse/out on spark1 into docs/results/2026-10-19-cuda-round6/raw/ (or DEST),
# small slot, vLLM up:   fleetq run --host spark1 --mem 0 --label "round6 fetch" -- scripts/cuda-round6/fetch.sh r6A [r6B ...]
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST=${DEST:-$REPO/docs/results/2026-10-19-cuda-round6/raw}
for d in "$@"; do
  mkdir -p "$DEST/$d"
  rsync -a --exclude '*.safetensors' --exclude '*.nsys-rep' --exclude '*.sqlite' --exclude 'kern_cuda_gpu_trace.csv' \
    -e "ssh -o BatchMode=yes" --rsync-path="FLEETQ_JOB=$FLEETQ_JOB rsync" "$HOST:intarsia-decfuse/out/$d/" "$DEST/$d/"
done
