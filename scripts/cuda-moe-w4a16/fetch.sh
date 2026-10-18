#!/usr/bin/env bash
# Copy a result directory of ~/intarsia-decfuse/out on spark1 into docs/results/2026-10-18-cuda-moe-w4a16/raw/ (small slot, vLLM up):
#   fleetq run --host spark1 --mem 1 --label "w4a16 fetch" -- scripts/cuda-moe-w4a16/fetch.sh w4A [w4B ...]
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for d in "$@"; do
  mkdir -p "$REPO/docs/results/2026-10-18-cuda-moe-w4a16/raw/$d"
  rsync -a --exclude '*.safetensors' --exclude 'serve.log' -e "ssh -o BatchMode=yes" --rsync-path="FLEETQ_JOB=$FLEETQ_JOB rsync" \
    "$HOST:intarsia-decfuse/out/$d/" "$REPO/docs/results/2026-10-18-cuda-moe-w4a16/raw/$d/"
done
