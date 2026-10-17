#!/usr/bin/env bash
# Copy a results directory of spark1 (~/intarsia-decfuse/out/<subdir>) into a local directory, without the big binary files
# (.safetensors dumps, .nsys-rep, .sqlite). Runs INSIDE a fleetq slot (no GPU work):
#   fleetq run --host spark1 --mem 0 --label "..." -- scripts/cuda-decode/pull-out.sh <subdir> <local dir>
# FLEETQ_JOB (set by `fleetq run`) goes on the ssh.
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
SUB=${1:?subdir under ~/intarsia-decfuse/out}; DEST=${2:?local dir}
HOST=${HOST:-spark1}
mkdir -p "$DEST"
ssh -n -o BatchMode=yes "$HOST" "FLEETQ_JOB=$FLEETQ_JOB tar -C ~/intarsia-decfuse/out/$SUB -cf - --exclude='*.safetensors' --exclude='*.nsys-rep' --exclude='*.sqlite' ." | tar -xf - -C "$DEST"
du -sh "$DEST"
