#!/bin/bash
# chunk32k window: ab0 = chunk 512, ab1 = chunk 2048, same binary (~/intarsia-dsv41-prefill/bin/intarsia), frozen question set in 3 shards
export SPARK_DIR=~/intarsia-dsv41-prefill
W=/opt/code/intarsiaml/.worktrees/chunk32k
cd $W
S=$W/.scratch
export LSHARDS="$S/items.pilot.jsonl $S/items.rest1.jsonl $S/items.rest2.jsonl"
export OUT=$S/win1 CAP=33000 STEPS="ldata_shards" LDATA_TOKENS=48 EARLYOOM_OK="3,2 2,1" MAX_MIN=105 GATE_WAIT=300
export AB="INTARSIA_DSV41_CHUNK=512|INTARSIA_DSV41_CHUNK=2048"
exec scripts/dsv41-tp2/run-full-chunk32k.sh
