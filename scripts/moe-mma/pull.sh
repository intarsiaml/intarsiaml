#!/usr/bin/env bash
# Copy result logs of spark1's ~/intarsia-decfuse/<tree>/out/<dir> into docs/results/2026-10-17-moe-mma-small-t/raw/<dir>
# (no GPU; run in a fleetq slot so the ssh carries FLEETQ_JOB):
#   fleetq run --host spark1 --mem 0 --label "..." -- scripts/moe-mma/pull.sh <tree> <dir>...
# e.g. pull.sh moe-mma-dev w1 w2. Files over 2 MB and safetensors dumps are not copied.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TREE=${1:?tree}; shift
for d in "$@"; do
  DEST=$REPO/docs/results/2026-10-17-moe-mma-small-t/raw/$d
  mkdir -p "$DEST"
  ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsia-decfuse/$TREE/out/$d && find . -type f -size -2M ! -name '*.safetensors'" | while read -r f; do
    mkdir -p "$DEST/$(dirname "$f")"
    scp -q -o BatchMode=yes "spark1:intarsia-decfuse/$TREE/out/$d/$f" "$DEST/$f" || true
  done
done
echo "pulled into $REPO/docs/results/2026-10-17-moe-mma-small-t/raw"
