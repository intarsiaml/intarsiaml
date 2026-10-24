#!/usr/bin/env bash
# Lane c8k: run a script from ~/intarsia-decfuse/pf-scripts on a Spark inside the current fleetq slot, after syncing scripts/c8k, the
# committed oracles and the derived reuse goldens to it (so a job queued before they changed still runs the current ones).
#   fleetq run --host spark1 ... -- scripts/c8k/rjob.sh spark1 c8k/gatesA.sh <args>
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
H=$1; S=$2; shift 2
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; G="$D/../../golden"
ssh -n -o BatchMode=yes "$H" "mkdir -p ~/intarsia-decfuse/pf-scripts/c8k ~/intarsia-decfuse/pf-golden/oracle ~/intarsia-decfuse/pf-golden/reuse-splits"
rsync -a "$D/" "$H:intarsia-decfuse/pf-scripts/c8k/"
rsync -a "$G/oracle/" "$H:intarsia-decfuse/pf-golden/oracle/"
rsync -a "$G/reuse-splits/" "$H:intarsia-decfuse/pf-golden/reuse-splits/"
exec ssh -n -o BatchMode=yes "$H" "export FLEETQ_JOB=$FLEETQ_JOB WAIT_MIN=${WAIT_MIN:-0}; bash ~/intarsia-decfuse/pf-scripts/$S $(printf '%q ' "$@")"
