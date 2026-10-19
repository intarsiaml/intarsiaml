#!/usr/bin/env bash
# Run a script of the synced tree on spark1 inside the caller's fleetq slot (FLEETQ_JOB goes on the ssh):
#   fleetq run --host spark1 --mem 45 --label "..." -- scripts/cuda-prefill4/remote.sh p4-dev scripts/cuda-prefill4/acc.sh
# Extra environment for the remote script: REMOTE_ENV="A=1 B=2".
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
TREE=${1:?tree}; SCRIPT=${2:?script path in the tree}; HOST=${HOST:-spark1}
ssh -n -o BatchMode=yes "$HOST" "export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH FLEETQ_JOB=$FLEETQ_JOB ${REMOTE_ENV:-}; bash ~/intarsia-decfuse/$TREE/$SCRIPT"
