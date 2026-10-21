#!/usr/bin/env bash
# Copy round-7 result directories of ~/intarsia-decfuse/out on spark1: logs into docs/results/2026-10-21-cuda-round7/raw/<dir>
# (no safetensors), self-golden dumps (*.safetensors) into $DUMPS/<dir> when DUMPS is set. Small slot, vLLM up:
#   DUMPS=<scratch dir> fleetq run --host spark1 --mem 0 --label "..." -- scripts/cuda-round7/fetch.sh r7A [r7B ...]
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST=${DEST:-$REPO/docs/results/2026-10-21-cuda-round7/raw}
for d in "$@"; do
  mkdir -p "$DEST/$d"
  rsync -a --exclude '*.safetensors' --exclude '*.nsys-rep' --exclude '*.sqlite' \
    -e "ssh -o BatchMode=yes" --rsync-path="FLEETQ_JOB=$FLEETQ_JOB rsync" "$HOST:intarsia-decfuse/out/$d/" "$DEST/$d/"
  if [[ -n "${DUMPS:-}" ]]; then
    mkdir -p "$DUMPS/$d"
    rsync -a --include '*.safetensors' --exclude '*' \
      -e "ssh -o BatchMode=yes" --rsync-path="FLEETQ_JOB=$FLEETQ_JOB rsync" "$HOST:intarsia-decfuse/out/$d/" "$DUMPS/$d/"
  fi
done
