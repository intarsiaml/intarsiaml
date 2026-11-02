#!/bin/bash
# Full-model A/B/A/B of the Engram replay switch, one both-Spark window. Under fleetq.
set -u
: "${FLEETQ_JOB:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"; R="$HERE/../../scripts/dsv41-tp2"
export OUT=$HERE/win3 MAX_MIN=${MAX_MIN:-32} EARLYOOM_OK="3,2 2,1"
export KNOCK_DIR=$HERE/../tfknock LDATA=$HERE/../tfknock/knock-rcp.jsonl
export XENV="INTARSIA_DSV41_SPEC=1 INTARSIA_DSV41_PROFILE=1000000"
export STEPS="golden ldata lspeed" STEPS_AB0="margin golden ldata lspeed"
export AB="X=0|INTARSIA_DSV41_ENGRAM_ONEGRAPH=1|X=0|INTARSIA_DSV41_ENGRAM_ONEGRAPH=1"
"$R/run-full.sh"
