#!/bin/bash
# B2: controls and the 32K prompt. 8K prompt at chunk 512 with other, equivalent kernel paths (HC_FUSED=0, F16W_ROWS=1, EXL3_SPLITK=0, CAP 32768),
# the 32K prompt at chunk 512 / 1024 / 2048, then the 8K prompt as one chunk of 8192 last (its workspace may hit the memfine line).
export SPARK_DIR=~/intarsia-dsv41-prefill
W=/opt/code/intarsiaml/.worktrees/dsv41-prefill
cd $W
P8=3,100,255,256,511,512,513,1023,1024,2047,2048,2049,4095,4096,7999
P32=3,100,511,512,1023,1024,2047,2048,4095,4096,8191,8192,16383,16384,32767
WHAT=embed,hc_attn,hc_ffn,indexer,sparse_attn,.ffn.gate,head
D='$HOME/intarsia-dsv41-prefill/dump'
mk() { echo "INTARSIA_DSV41_CHUNK=$1 INTARSIA_DSV41_CAP=$4 $3 INTARSIA_DSV41_DUMP=$D/$2 INTARSIA_DSV41_DUMP_POS=$5 INTARSIA_DSV41_DUMP_WHAT=$WHAT"; }
L32=$W/docs/results/2026-11-02-dsv41-prefill/prompts/long32k.txt
AB="$(mk 512 r512 INTARSIA_NOOP=0 8300 $P8)|$(mk 512 hc0 INTARSIA_DSV41_HC_FUSED=0 8300 $P8)|$(mk 512 f16w1 INTARSIA_DSV41_F16W_ROWS=1 8300 $P8)|$(mk 512 splitk0 INTARSIA_DSV41_EXL3_SPLITK=0 8300 $P8)|$(mk 512 cap32k INTARSIA_NOOP=0 32768 $P8)|$(mk 512 l512 INTARSIA_NOOP=0 32768 $P32)|$(mk 1024 l1024 INTARSIA_NOOP=0 32768 $P32)|$(mk 2048 l2048 INTARSIA_NOOP=0 32768 $P32)|$(mk 8192 c8192 INTARSIA_NOOP=0 8300 $P8)"
for i in 0 1 2 3 4 5 6 7 8; do export BIN_AB$i='$HOME/intarsia-dsv41-prefill/bin/intarsia.dump'; done
for i in 5 6 7; do export LONG_PROMPTS_AB$i=$L32; done
OUT=$W/.scratch/winB2 CAP=8300 STEPS="longp" LONG_PROMPTS=$W/docs/results/2026-11-02-dsv41-prefill/prompts/doc8k.txt EARLYOOM_OK="3,2 2,1" AB="$AB" MAX_MIN=28 exec scripts/dsv41-tp2/run-full.sh
