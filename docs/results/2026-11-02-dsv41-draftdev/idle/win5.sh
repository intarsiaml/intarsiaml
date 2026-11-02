#!/bin/bash
# Full-model A/B/A/B of the remote rank's polling, golden on both B configs, nsys on a B config. Under fleetq, both Sparks.
set -u
: "${FLEETQ_JOB:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"; R="$HERE/../../scripts/dsv41-tp2"
export OUT=$HERE/win5 MAX_MIN=${MAX_MIN:-22} EARLYOOM_OK="3,2 2,1"
export LDATA=$HERE/../tfknock/knock-rcp.jsonl NSYS_PROMPT_FILE=$HERE/../tfprof/prompt.txt
export XENV="INTARSIA_DSV41_SPEC=1 INTARSIA_DSV41_PROFILE=1000000"
export STEPS="ldata lspeed" STEPS_AB1="golden ldata lspeed" STEPS_AB3="golden ldata lspeed" STEPS_AB4="nsys" NSYS_AB4=1
export AB="X=0|INTARSIA_TP_SPIN_MS=100|X=0|INTARSIA_TP_SPIN_MS=100|INTARSIA_TP_SPIN_MS=100"
"$R/run-full.sh"
