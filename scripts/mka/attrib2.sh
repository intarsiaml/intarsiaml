#!/bin/sh
# Lane mka attribution, part 2: is the extra row memory or in-core work?
set -u
OUT=${1:?outdir}
BIN=${BIN:-target/release/examples/attn_bw}
mkdir -p "$OUT"
S35="--heads 16 --kv-heads 2 --dim 256 --layers 10"
S27="--heads 24 --kv-heads 4 --dim 256 --layers 16"
run() { name=$1; shift; echo "== $name" | tee -a "$OUT/attrib.txt"; env "$@" 2>&1 | grep -v "^pso" | tee -a "$OUT/attrib.txt"; }
REPS="--reps ${REPS:-30} --warm 3"
# E1: rows adjacent in the grid (G0b verify kernel) against the row-table kernel on the same cache
run "E1 35B rows-adjacent vs rowtab" $BIN $S35 $REPS --rows 1,2,3,8 rows rowtab 4096 34000
run "E1 27B rows-adjacent vs rowtab" $BIN $S27 $REPS --rows 1,2,3,8 rows rowtab 4096 34000
# E2: slope per key and row on ONE small hot cache (the bytes are in the GPU caches for sure at 256 / 1024 keys)
run "E2 35B hot small" $BIN $S35 $REPS --hot --rows 1,2,4,8,16 rowtab 256 1024 4096
run "E2 27B hot small" $BIN $S27 $REPS --hot --rows 1,2,4,8,16 rowtab 256 1024 4096
# E4: threadgroup shape at the same simdgroup work (27B: the plain head split HS=3 has 120 threadgroups of 4 simdgroups,
# the bitwise one 360 of 2)
run "E4 27B HS=3 (plain split, 4 simdgroups)" INTARSIA_METAL_ATTN_HS=3 $BIN $S27 $REPS --rows 1,3,8 splitkv rowtab 4096 34000
run "E4 27B HS=6" INTARSIA_METAL_ATTN_HS=6 $BIN $S27 $REPS --rows 1,3,8 splitkv rowtab 34000
run "E4 35B HS=2" INTARSIA_METAL_ATTN_HS=2 $BIN $S35 $REPS --rows 1,3,8 splitkv rowtab 4096 34000
# E5: split policy at the same work: fewer, longer threadgroups / more, shorter ones
run "E5 35B target 60"  INTARSIA_METAL_SPLITKV=128,60  $BIN $S35 $REPS --rows 1,3,8 splitkv rowtab 34000
run "E5 35B target 240" INTARSIA_METAL_SPLITKV=128,240 $BIN $S35 $REPS --rows 1,3,8 splitkv rowtab 34000
run "E5 27B target 60"  INTARSIA_METAL_SPLITKV=128,60  $BIN $S27 $REPS --rows 1,3,8 splitkv rowtab 34000
run "E5 27B target 240" INTARSIA_METAL_SPLITKV=128,240 $BIN $S27 $REPS --rows 1,3,8 splitkv rowtab 34000
echo done >> "$OUT/attrib.txt"
