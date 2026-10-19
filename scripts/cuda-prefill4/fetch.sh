#!/usr/bin/env bash
# Copy result directories of ~/intarsia-decfuse/out on spark1 into docs/results/2026-10-19-cuda-prefill-4/raw/ (small slot, vLLM up):
#   fleetq run --host spark1 --mem 1 --label "prefill-4 fetch" -- scripts/cuda-prefill4/fetch.sh p4A [p4B ...]
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for d in "$@"; do
  mkdir -p "$REPO/docs/results/2026-10-19-cuda-prefill-4/raw/$d"
  rsync -a --exclude '*.safetensors' --exclude '*.nsys-rep' --exclude '*.sqlite' -e "ssh -o BatchMode=yes" --rsync-path="FLEETQ_JOB=$FLEETQ_JOB rsync" \
    "$HOST:intarsia-decfuse/out/$d/" "$REPO/docs/results/2026-10-19-cuda-prefill-4/raw/$d/"
done
