#!/bin/bash
# compile the repo's kernels the bench needs (the same flags as crates/intarsia-backend-cuda/build.rs) and run it with each env setting.
# usage (on a Spark, inside a fleetq slot): build-run.sh <repo src dir>
set -eu
SRC=${1:?repo src}; B=$(cd "$(dirname "$0")" && pwd); OUT=${OUT:-$HOME/intarsia-dsv41-attnfuse/bench}; mkdir -p "$OUT"
export PATH=/usr/local/cuda/bin:$PATH
FL="-c -O3 --std=c++17 -lineinfo --expt-relaxed-constexpr -arch=sm_121 -I $SRC/kernels/cuda -I $SRC/kernels"
for f in dsv41_misc dsv41_hc dsv41_hc_sub dsv41_attn; do nvcc $FL $SRC/kernels/cuda/$f.cu -o $OUT/$f.o & done
wait
nvcc -O3 --std=c++17 -arch=sm_121 $B/hc_rope_bench.cu $OUT/dsv41_misc.o $OUT/dsv41_hc.o $OUT/dsv41_hc_sub.o $OUT/dsv41_attn.o -o $OUT/hc_rope_bench
for e in "0" "1"; do INTARSIA_DSV41_ROPE_VEC=$e INTARSIA_DSV41_HC_VEC=$e INTARSIA_DSV41_ATTN_PF=$e $OUT/hc_rope_bench ${ITERS:-20} | tee $OUT/bench-vec$e.txt; done
echo "--- hash diff between VEC=0 and VEC=1 (empty = bitwise identical):"
P='hash|^hc S=.* (sep|fold) +h '
diff <(grep -E "$P" $OUT/bench-vec0.txt | sed 's/ *[0-9.]* ms.*//') <(grep -E "$P" $OUT/bench-vec1.txt | sed 's/ *[0-9.]* ms.*//') && echo IDENTICAL
echo "--- sep vs fold within the VEC=1 run (each pair of lines must agree on every field after the tag; printed MISMATCH otherwise):"
grep -E "^hc S=.* (sep|fold) +h " $OUT/bench-vec1.txt | sed -E 's/ (sep|fold) +h / h /' | awk '{k=$2" "$3" "$4; v=$0; sub(/^hc S=[^ ]* D=[^ ]* T=[^ ]* /,"",v); if (k in a) { print (a[k]==v ? "match " : "MISMATCH ") k } else a[k]=v }'
