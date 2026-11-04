#!/bin/bash
# chunk32k-2 pilot: ab0 = chunk 512 on the pilot items (difficulty check), ab1 = chunk 2048 on the 65K pilot items only (memory check)
export SPARK_DIR=~/intarsia-dsv41-prefill
W=/opt/code/intarsiaml/.worktrees/chunk32k-2
cd $W
S=$W/.scratch
export LSHARDS="$S/pilot.s32-00.jsonl $S/pilot.s65-00.jsonl"
export LSHARDS_AB1="$S/pilot.s65-00.jsonl"
export OUT=$S/pilot-win CAP=66000 STEPS="ldata_shards" LDATA_TOKENS=160 EARLYOOM_OK="3,2 2,1" MAX_MIN=62
export AB="INTARSIA_DSV41_CHUNK=512|INTARSIA_DSV41_CHUNK=2048"
exec scripts/dsv41-tp2/run-full-chunk32k2.sh
