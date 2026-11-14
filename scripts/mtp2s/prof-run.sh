#!/usr/bin/env bash
# mtp2s: copy the lane's scripts to the Spark and run prof.sh's arms there (under fleetq run --quiet gpu,cpu).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
scp -q -o BatchMode=yes "$S"/*.py "$S"/*.sh spark2:intarsia-mtp2s/src/scripts/mtp2s/
ssh -o BatchMode=yes spark2 "export FLEETQ_JOB=$FLEETQ_JOB; bash -s $*" < "$S/prof.sh"
