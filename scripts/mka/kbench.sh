#!/bin/sh
# Lane mka kernel bench: packed (INTARSIA_METAL_ATTN_KV_ONCE=1) against unpacked, fresh process per arm, ABBA.
# usage: kbench.sh <outdir> [reps]
set -u
OUT=${1:?outdir}
BIN=${BIN:-target/release/examples/attn_bw}
mkdir -p "$OUT"
S35="--heads 16 --kv-heads 2 --dim 256"
S27="--heads 24 --kv-heads 4 --dim 256"
REPS="--reps ${2:-20} --warm 3"
CTX=${CTX:-"4096 16384 34000"}
run() { name=$1; shift; echo "== $name" | tee -a "$OUT/kbench.txt"; env "$@" 2>&1 | grep -v "^pso\|^heads\| visible" | cut -c1-110 | tee -a "$OUT/kbench.txt"; }
for arm in off on on off; do
  K=0; [ $arm = on ] && K=1
  # chained launches (one cache per layer)
  run "35B chained kvo=$arm" INTARSIA_METAL_ATTN_KV_ONCE=$K $BIN $S35 --layers 10 --cap 40000 $REPS --rows 1,3,8 splitkv rows rowtab $CTX
  run "27B chained kvo=$arm" INTARSIA_METAL_ATTN_KV_ONCE=$K $BIN $S27 --layers 16 --cap 40000 $REPS --rows 1,3,8 splitkv rows rowtab $CTX
  # a 32 MB copy between the launches (in-engine-like)
  run "35B gap32 kvo=$arm" INTARSIA_METAL_ATTN_KV_ONCE=$K $BIN $S35 --layers 10 --cap 40000 --gap 32 $REPS --rows 1,3,8 splitkv rows rowtab $CTX
  run "27B gap32 kvo=$arm" INTARSIA_METAL_ATTN_KV_ONCE=$K $BIN $S27 --layers 16 --cap 40000 --gap 32 $REPS --rows 1,3,8 splitkv rows rowtab $CTX
done
echo done >> "$OUT/kbench.txt"
