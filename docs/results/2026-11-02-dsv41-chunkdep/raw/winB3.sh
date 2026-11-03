#!/bin/bash
# B3: the 32K prompt at chunk 512 / 1024 / 2048 (B2 ran with CAP 32768: "the prompt has 32768 tokens but the context window is 32768 tokens", no dumps).

export SPARK_DIR=~/intarsia-dsv41-prefill
W=/opt/code/intarsiaml/.worktrees/dsv41-prefill
cd $W
P8=3,100,255,256,511,512,513,1023,1024,2047,2048,2049,4095,4096,7999
P32=3,100,511,512,1023,1024,2047,2048,4095,4096,8191,8192,16383,16384,32767
WHAT=embed,hc_attn,hc_ffn,indexer,sparse_attn,.ffn.gate,head
D='$HOME/intarsia-dsv41-prefill/dump'
mk() { echo "INTARSIA_DSV41_CHUNK=$1 INTARSIA_DSV41_CAP=$4 $3 INTARSIA_DSV41_DUMP=$D/$2 INTARSIA_DSV41_DUMP_POS=$5 INTARSIA_DSV41_DUMP_WHAT=$WHAT"; }
L32=$W/docs/results/2026-11-02-dsv41-prefill/prompts/long32k.txt
AB="$(mk 512 l512 INTARSIA_NOOP=0 33000 $P32)|$(mk 1024 l1024 INTARSIA_NOOP=0 33000 $P32)|$(mk 2048 l2048 INTARSIA_NOOP=0 33000 $P32)"
for i in 0 1 2; do export BIN_AB$i='$HOME/intarsia-dsv41-prefill/bin/intarsia.dump'; done
for i in 0 1 2; do export LONG_PROMPTS_AB$i=$L32; done
OUT=$W/.scratch/winB3 CAP=33000 STEPS="longp" LONG_PROMPTS=$W/docs/results/2026-11-02-dsv41-prefill/prompts/doc8k.txt EARLYOOM_OK="3,2 2,1" AB="$AB" MAX_MIN=22 exec scripts/dsv41-tp2/run-full.sh
