#!/usr/bin/env bash
# Lane pf: run a script from ~/intarsia-decfuse/pf-scripts on a Spark inside the current fleetq slot.
#   fleetq run --host spark1 ... -- scripts/pf/rjob.sh spark1 pf/prof.sh <args>
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
H=$1; S=$2; shift 2
exec ssh -n -o BatchMode=yes "$H" "export FLEETQ_JOB=$FLEETQ_JOB; bash ~/intarsia-decfuse/pf-scripts/$S $(printf '%q ' "$@")"
