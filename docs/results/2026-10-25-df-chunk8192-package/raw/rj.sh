#!/usr/bin/env bash
# run ~/intarsia-decfuse/df-scripts/<script> on spark2 inside the current fleetq slot
set -uo pipefail
: "${FLEETQ_JOB:?}"
S=$1; shift
exec ssh -n -o BatchMode=yes spark2 "export FLEETQ_JOB=$FLEETQ_JOB; bash ~/intarsia-decfuse/df-scripts/$S $(printf '%q ' "$@")"
