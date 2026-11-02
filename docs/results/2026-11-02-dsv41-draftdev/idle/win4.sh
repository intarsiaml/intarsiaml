#!/bin/bash
# Final head: 3-layer gates (default vs off, TP=2 over NCCL; CUDA bitwise test), then the full-model window: golden twice, A/B against the
# old segments, nsys on the prompt the TF trace used. Under fleetq, both Sparks.
set -u
: "${FLEETQ_JOB:?}"
HERE="$(cd "$(dirname "$0")" && pwd)"; R="$HERE/../../scripts/dsv41-tp2"
ARMS='on=-;off=INTARSIA_DSV41_ENGRAM_ONEGRAPH=0;on2=-;off2=INTARSIA_DSV41_ENGRAM_ONEGRAPH=0' OUT=$HERE/out4 "$HERE/run3.sh"
C='INTARSIA_DSV41_BACKEND=cuda INTARSIA_DSV41_CACHE=$HOME/intarsia-dsv41-tp2/cache/l3-r0of1 INTARSIA_DSV41_DP_SPEC=1 INTARSIA_DSV41_DPOS_GRAPHS=1'
for i in 1 2; do "$R/spark-test.sh" spark1 dsv41_runtime device_position_passes_equal_host_passes $C || { echo "device test failed"; exit 1; }; done
export OUT=$HERE/win4 MAX_MIN=${MAX_MIN:-24} EARLYOOM_OK="3,2 2,1"
export KNOCK_DIR=$HERE/../tfknock LDATA=$HERE/../tfknock/knock-rcp.jsonl NSYS_PROMPT_FILE=$HERE/../tfprof/prompt.txt
export XENV="INTARSIA_DSV41_SPEC=1 INTARSIA_DSV41_PROFILE=1000000"
export STEPS="golden ldata lspeed" STEPS_AB1="ldata lspeed" STEPS_AB3="nsys" NSYS_AB3=1
export AB="X=0|INTARSIA_DSV41_ENGRAM_ONEGRAPH=0|X=0|X=1"
"$R/run-full.sh"
