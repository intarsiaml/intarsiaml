#!/bin/bash
# chunk32k-2 main window: ab0 = chunk 512, ab1 = chunk 2048, ab2 = chunk 512 with HC_FUSED=0 (noise control, last); stop by 06:30 EST on 11-04
export SPARK_DIR=~/intarsia-dsv41-prefill
W=/opt/code/intarsiaml/.worktrees/chunk32k-2
cd $W
S=$W/.scratch
for i in $(seq 1 60); do [ -f $S/FROZEN ] && break; sleep 10; done     # the set is frozen (committed) by the coordinator-visible marker
[ -f $S/FROZEN ] || { echo "no $S/FROZEN: set not frozen, not running"; exit 1; }
export LSHARDS="$(cat $S/shards.txt)"
export STOP_EPOCH=$(date -j -f "%Y-%m-%d %H:%M" "2026-11-04 06:30" +%s)
export OUT=$S/main-win CAP=66000 STEPS="ldata_shards" LDATA_TOKENS=160 EARLYOOM_OK="3,2 2,1"
export AB="INTARSIA_DSV41_CHUNK=512|INTARSIA_DSV41_CHUNK=2048|INTARSIA_DSV41_CHUNK=512 INTARSIA_DSV41_HC_FUSED=0"
exec scripts/dsv41-tp2/run-full-chunk32k2.sh
