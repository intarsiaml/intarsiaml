#!/usr/bin/env bash
# Lane sp2: per-op prefill profile of the base tree (origin/main) on spark1, 4k + 16k, inside an exclusive fleetq slot:
#   fleetq run --host spark1 --exclusive --mem 40 --label "..." -- scripts/sp2/prof-job.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
HOST=${HOST:-spark1}
ssh -n -o BatchMode=yes "$HOST" "export FLEETQ_JOB=$FLEETQ_JOB STEPS=prof OUTNAME=${OUTNAME:-sp2P} ARMS='${ARMS:-base:sp2-base}' CHARS='${CHARS:-14000 55000}' GOLDEN_DIR=\$HOME/intarsiaml-golden; bash ~/intarsia-decfuse/sp2-dev/scripts/sp2/remote.sh"
