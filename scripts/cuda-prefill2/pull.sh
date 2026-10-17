#!/usr/bin/env bash
# Copy result files of spark1's ~/intarsia-decfuse/out into docs/results/2026-10-17-cuda-prefill-2/raw (no GPU; run in a
# fleetq slot so the ssh carries FLEETQ_JOB):  fleetq run --host spark1 --mem 0 --label "..." -- scripts/cuda-prefill2/pull.sh <glob>...
# e.g. pull.sh 'w1/*'. Large traces (.nsys-rep, .sqlite, safetensors) are not copied.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST=$REPO/docs/results/2026-10-17-cuda-prefill-2/raw
mkdir -p "$DEST"
for g in "$@"; do
  ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsia-decfuse/out && find $g -type f -size -2M ! -name '*.nsys-rep' ! -name '*.sqlite' ! -name '*.safetensors'" | while read -r f; do
    mkdir -p "$DEST/$(dirname "$f")"
    scp -q -o BatchMode=yes "spark1:intarsia-decfuse/out/$f" "$DEST/$f" || true
  done
done
echo "pulled into $DEST"
