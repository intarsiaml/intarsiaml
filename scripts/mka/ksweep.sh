#!/bin/sh
# Lane mka: threadgroup size sweep of the packed kernels (windows per threadgroup, trim on / off). usage: ksweep.sh <outdir>
set -u
OUT=${1:?outdir}
BIN=${BIN:-target/release/examples/attn_bw}
mkdir -p "$OUT"
S35="--heads 16 --kv-heads 2 --dim 256 --layers 10"
S27="--heads 24 --kv-heads 4 --dim 256 --layers 16"
REPS="--reps ${2:-15} --warm 3"
run() { name=$1; shift; echo "== $name" | tee -a "$OUT/ksweep.txt"; env "$@" 2>&1 | grep -v "^pso\|^heads\| visible\|^gap" | cut -c1-110 | tee -a "$OUT/ksweep.txt"; }
K="splitkv rows rowtab"
for gap in 0 32; do
  G=""; [ $gap != 0 ] && G="--gap $gap"
  run "35B gap$gap unpacked" INTARSIA_METAL_ATTN_KV_ONCE=0 $BIN $S35 --cap 40000 $G $REPS --rows 1,3,8 $K 4096 34000
  run "27B gap$gap unpacked" INTARSIA_METAL_ATTN_KV_ONCE=0 $BIN $S27 --cap 40000 $G $REPS --rows 1,3,8 $K 4096 34000
  for ipt in 1 2 3 4 6; do for trim in 1 0; do
    [ $trim = 0 ] && [ $ipt != 1 ] && [ $ipt != 3 ] && continue
    run "35B gap$gap ipt=$ipt trim=$trim" INTARSIA_METAL_ATTN_KV_ONCE=1 INTARSIA_METAL_ATTN_KVO_IPT=$ipt INTARSIA_METAL_ATTN_KVO_TRIM=$trim $BIN $S35 --cap 40000 $G $REPS --rows 1,3,8 $K 4096 34000
    run "27B gap$gap ipt=$ipt trim=$trim" INTARSIA_METAL_ATTN_KV_ONCE=1 INTARSIA_METAL_ATTN_KVO_IPT=$ipt INTARSIA_METAL_ATTN_KVO_TRIM=$trim $BIN $S27 --cap 40000 $G $REPS --rows 1,3,8 $K 4096 34000
  done; done
done
# threadgroup memory: the same layouts with the largest accumulator instance (does the reserved memory cost concurrency?)
for ipt in 1 3; do
  run "35B gap32 ipt=$ipt trim=1 tgf=7680" INTARSIA_METAL_ATTN_KV_ONCE=1 INTARSIA_METAL_ATTN_KVO_TGF=7680 INTARSIA_METAL_ATTN_KVO_IPT=$ipt $BIN $S35 --cap 40000 --gap 32 $REPS --rows 1,3,8 $K 4096 34000
  run "27B gap32 ipt=$ipt trim=1 tgf=7680" INTARSIA_METAL_ATTN_KV_ONCE=1 INTARSIA_METAL_ATTN_KVO_TGF=7680 INTARSIA_METAL_ATTN_KVO_IPT=$ipt $BIN $S27 --cap 40000 --gap 32 $REPS --rows 1,3,8 $K 4096 34000
done
echo done >> "$OUT/ksweep.txt"
