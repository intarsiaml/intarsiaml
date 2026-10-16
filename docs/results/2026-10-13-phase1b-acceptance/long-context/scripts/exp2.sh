#!/usr/bin/env bash
set -u
D=$HOME/intarsia-dbg; BIN=$D/src/target/dbg/release/intarsia
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
G=$D/g/tiny.safetensors
SIMT=attn_decode_bf16_splitkv,attn_prefill_bf16_splitkv,attn_prefill_bf16_flash
COMMON="--model $M --backend cuda --max-seq 8192 --mem-budget 70 --mode prefill --prefill-chunk 256 --allow-unstamped --allow-dirty --allow-same-code"
cd $D/out; rm -rf tr_flash tr_simt
echo "== flash trace"; date +%T
INTARSIA_TRACE_ATTN=$D/out/tr_flash/a INTARSIA_TRACE_ATTN_MAXLEN=400 INTARSIA_TRACE_BLK=$D/out/tr_flash/b $BIN accuracy $COMMON --golden $G > log2_flash.txt 2>&1; tail -8 log2_flash.txt | cut -c1-180
echo "== simt trace"; date +%T
INTARSIA_CUDA_DISABLE=$SIMT INTARSIA_TRACE_ATTN=$D/out/tr_simt/a INTARSIA_TRACE_ATTN_MAXLEN=400 INTARSIA_TRACE_BLK=$D/out/tr_simt/b $BIN accuracy $COMMON --golden $G > log2_simt.txt 2>&1; tail -8 log2_simt.txt | cut -c1-180
date +%T; ls tr_flash/a | wc -l; ls tr_flash/b | wc -l; du -sh tr_flash tr_simt
