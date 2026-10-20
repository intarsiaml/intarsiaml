#!/usr/bin/env bash
# Copy this checkout's scripts/ to spark1:~/intarsia-decfuse/r6-scripts/scripts/ (no build: the built trees stay as they
# are), then optionally run one of them there with REMOTE_ENV, inside the caller's fleetq slot:
#   REMOTE_ENV="TREE=r6-base" fleetq run ... -- scripts/cuda-round6/sync-scripts.sh scripts/cuda-round6/prof-prefill.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ssh -n -o BatchMode=yes "$HOST" "FLEETQ_JOB=$FLEETQ_JOB mkdir -p intarsia-decfuse/r6-scripts"
rsync -a --delete -e "ssh -o BatchMode=yes" --rsync-path="FLEETQ_JOB=$FLEETQ_JOB rsync" "$REPO/scripts/" "$HOST:intarsia-decfuse/r6-scripts/scripts/"
[[ $# -ge 1 ]] || exit 0
ssh -n -o BatchMode=yes "$HOST" "export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH FLEETQ_JOB=$FLEETQ_JOB ${REMOTE_ENV:-}; bash ~/intarsia-decfuse/r6-scripts/$1"
