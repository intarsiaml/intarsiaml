#!/usr/bin/env bash
# Copy files into a tree on spark1 (a window script that was written after the tree was built).
# Run INSIDE a fleetq slot:  fleetq run --host spark1 --mem 1 --label "..." -- scripts/cuda-prefill2/push.sh <tree> <file>...
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
TREE=${1:?tree name}; shift
HOST=${HOST:-spark1}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
for f in "$@"; do
  d=$(dirname "$f")
  ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB; mkdir -p ~/intarsia-decfuse/$TREE/$d"
  scp -q -o BatchMode=yes "$REPO/$f" "$HOST:intarsia-decfuse/$TREE/$f"
  echo "pushed $f -> $TREE"
done
