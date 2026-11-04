#!/bin/bash
# Build the attn_sparse_mma A/B bench on a Spark: unmodified kernel (base) and the ATTN_2BLK variant (attn-2blk.patch applied to a copy of the
# repo file), each with the occupancy probe appended, same nvcc flags as crates/intarsia-backend-cuda/build.rs. Run inside a fleetq slot (--load cpu).
# usage: ab-build.sh <repo src dir>   (the Spark's ~/intarsia-dsv41-attnfuse/src is main 47af18777; its dsv41_attn.cu is the branch's)
set -eu
SRC=${1:?repo src}; B=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:-$HOME/intarsia-dsv41-attnfuse/bench/ab}; mkdir -p "$OUT"
export PATH=/usr/local/cuda/bin:$PATH
cp "$SRC/kernels/cuda/dsv41_attn.cu" "$OUT/attn_base_src.cu"
cp "$SRC/kernels/cuda/dsv41_attn.cu" "$OUT/attn_var_src.cu"
patch "$OUT/attn_var_src.cu" < "$B/attn-2blk.patch"
cat "$OUT/attn_base_src.cu" "$B/occ_probe.inc" > "$OUT/attn_base.cu"
cat "$OUT/attn_var_src.cu" "$B/occ_probe.inc" > "$OUT/attn_var.cu"
sed 's#"../../../../../kernels/abi/ik_abi.h"#"abi/ik_abi.h"#' "$B/attn_ab_bench.cu" > "$OUT/attn_ab_bench.cu"
FL="-c -O3 --std=c++17 -lineinfo --expt-relaxed-constexpr -arch=sm_121 -I $SRC/kernels/cuda -I $SRC/kernels"
nvcc $FL "$OUT/attn_base.cu" -o "$OUT/attn_base.o" &
nvcc $FL -DATTN_2BLK "$OUT/attn_var.cu" -o "$OUT/attn_var.o" &
wait
M=$HOME/intarsia-dsv41-attnfuse/bench
for v in base var; do
  nvcc -O3 --std=c++17 -arch=sm_121 -I $SRC/kernels "$OUT/attn_ab_bench.cu" $M/dsv41_misc.o $M/dsv41_hc.o $M/dsv41_hc_sub.o "$OUT/attn_$v.o" -o "$OUT/ab_$v"
  echo "== $v: cuobjdump -res-usage (attn_sparse_mma)"
  cuobjdump -res-usage "$OUT/attn_$v.o" 2>&1 | grep -A1 "attn_sparse_mma" | head -4
done
