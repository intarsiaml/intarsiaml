#!/usr/bin/env bash
# Copy result files of spark1's ~/intarsia-decfuse/out into docs/results/2026-10-14-cuda-decode-fusion/ (no GPU; run in a
# fleetq slot so the ssh carries FLEETQ_JOB):  fleetq run --host spark1 --mem 0 --label "..." -- scripts/cuda-decode/pull.sh <subdir-glob>...
# e.g. pull.sh 'prof-dev/steps.*' 'prof-*/nsys.out' 'ab/*' 'w3/*'. Large traces (.nsys-rep, .sqlite, trace csv) are not copied.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST=$REPO/docs/results/2026-10-14-cuda-decode-fusion/raw
mkdir -p "$DEST"
for g in "$@"; do
  ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsia-decfuse/out && find $g -type f -size -2M ! -name '*.nsys-rep' ! -name '*.sqlite' ! -name 'trace_*.csv' ! -name '*.safetensors'" | while read -r f; do
    mkdir -p "$DEST/$(dirname "$f")"
    scp -q -o BatchMode=yes "spark1:intarsia-decfuse/out/$f" "$DEST/$f" || true
  done
done
echo "pulled into $DEST"
