#!/bin/sh
# Lane mka attribution, part 3: why a launch costs more in the engine than in the chained microbench, and the prefill kernel the
# MTP head's catch-up rows run today.
set -u
OUT=${1:?outdir}
BIN=${BIN:-target/release/examples/attn_bw}
mkdir -p "$OUT"
S35="--heads 16 --kv-heads 2 --dim 256"
S27="--heads 24 --kv-heads 4 --dim 256"
run() { name=$1; shift; echo "== $name" | tee -a "$OUT/attrib.txt"; env "$@" 2>&1 | grep -v "^pso" | tee -a "$OUT/attrib.txt"; }
REPS="--reps ${REPS:-30} --warm 3"
# one launch per command buffer (as the per-dispatch trace of the engine measures it), cache capacity of the engine
run "35B isolated (1 launch per buffer), cap 40000" $BIN $S35 --layers 1 --cap 40000 $REPS --rows 1,3 splitkv rows rowtab 4096 32768
run "27B isolated (1 launch per buffer), cap 40000" $BIN $S27 --layers 1 --cap 40000 $REPS --rows 1,3 splitkv rows rowtab 4096 32768
run "35B chained 10, cap 40000" $BIN $S35 --layers 10 --cap 40000 $REPS --rows 1,3 splitkv rows rowtab 4096 32768
run "27B chained 16, cap 40000" $BIN $S27 --layers 16 --cap 40000 $REPS --rows 1,3 splitkv rows rowtab 4096 32768
# the prefill kernel on 2 / 3 / 8 rows at the end of a long cache (what the MTP head's catch-up launches today)
run "35B flash nax on M rows" $BIN $S35 --layers 4 --cap 40000 --reps 10 --warm 2 --rows 2,3,8 flashnax 4096 32768
run "27B flash nax on M rows" $BIN $S27 --layers 4 --cap 40000 --reps 10 --warm 2 --rows 2,3,8 flashnax 4096 32768
echo done >> "$OUT/attrib.txt"
