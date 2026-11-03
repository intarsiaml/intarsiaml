#!/bin/bash
# B1: dumps of one 8000-token prompt at chunk 512 / 1024 / 2048 / 4096 (bf16), 512 / 2048 (INTARSIA_DSV41_ACT=f32), 8192 last (its workspace may hit the 5 GiB memfine line)
export SPARK_DIR=~/intarsia-dsv41-prefill
W=/opt/code/intarsiaml/.worktrees/dsv41-prefill
cd $W
POS=3,100,255,256,511,512,513,1023,1024,2047,2048,2049,4095,4096,7999
WHAT=embed,hc_attn,hc_ffn,indexer,sparse_attn,.ffn.gate,head
D='$HOME/intarsia-dsv41-prefill/dump'
mk() { echo "INTARSIA_DSV41_CHUNK=$1 $3 INTARSIA_DSV41_DUMP=$D/$2 INTARSIA_DSV41_DUMP_POS=$POS INTARSIA_DSV41_DUMP_WHAT=$WHAT"; }
AB="$(mk 512 c512)|$(mk 1024 c1024)|$(mk 2048 c2048)|$(mk 4096 c4096)|$(mk 512 f32c512 INTARSIA_DSV41_ACT=f32)|$(mk 2048 f32c2048 INTARSIA_DSV41_ACT=f32)|$(mk 8192 c8192)"
export BIN_AB0='$HOME/intarsia-dsv41-prefill/bin/intarsia.dump'
for i in 1 2 3 4 5 6; do export BIN_AB$i='$HOME/intarsia-dsv41-prefill/bin/intarsia.dump'; done
OUT=$W/.scratch/winB1 CAP=8300 STEPS="longp" LONG_PROMPTS=$W/docs/results/2026-11-02-dsv41-prefill/prompts/doc8k.txt EARLYOOM_OK="3,2 2,1" AB="$AB" MAX_MIN=28 exec scripts/dsv41-tp2/run-full.sh
