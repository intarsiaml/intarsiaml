#!/bin/sh
# Lane mka attribution: decode attention at the two real shapes, visible 4k / 16k / 34k, M = 1 / 3 / 8.
# Run under fleetq (--quiet gpu). usage: attrib.sh <outdir> [extra env assignments are taken from the environment]
set -u
OUT=${1:?outdir}
BIN=${BIN:-target/release/examples/attn_bw}
mkdir -p "$OUT"
CTX="4096 16384 34000"
S35="--heads 16 --kv-heads 2 --dim 256 --layers 10"
S27="--heads 24 --kv-heads 4 --dim 256 --layers 16"
run() { # name, env..., -- args
  name=$1; shift
  echo "== $name" | tee -a "$OUT/attrib.txt"
  env "$@" 2>&1 | tee -a "$OUT/attrib.txt"
}
REPS="--reps ${REPS:-20} --warm 3"
# 1. today's launches, cold caches (one per layer)
run "35B default cold"  $BIN $S35 $REPS --rows 1,3,8 splitkv rowtab $CTX
run "27B default cold"  $BIN $S27 $REPS --rows 1,3,8 splitkv rowtab $CTX
# 2. the same on one hot cache (the compute side where the bytes fit the GPU caches)
run "35B default hot"   $BIN $S35 $REPS --hot --rows 1,3,8 splitkv rowtab $CTX
run "27B default hot"   $BIN $S27 $REPS --hot --rows 1,3,8 splitkv rowtab $CTX
# 3. head split variants (one row): how much the head-group re-read costs
for hs in 1 2 8; do
  run "35B HS=$hs cold" INTARSIA_METAL_ATTN_HS=$hs $BIN $S35 $REPS splitkv $CTX
done
for hsb in 0 2 6; do
  run "27B HSB=$hsb cold" INTARSIA_METAL_ATTN_HSB=$hsb $BIN $S27 $REPS splitkv $CTX
done
# 4. batched rows on their own caches
run "35B distinct cold" $BIN $S35 --layers 4 $REPS --rows 1,3,8 rowtabd $CTX
run "27B distinct cold" $BIN $S27 --layers 4 $REPS --rows 1,3,8 rowtabd $CTX
# 5. device position (token feedback) one row
run "35B devpos cold"   $BIN $S35 $REPS --devpos splitkv $CTX
run "27B devpos cold"   $BIN $S27 $REPS --devpos splitkv $CTX
echo done >> "$OUT/attrib.txt"
