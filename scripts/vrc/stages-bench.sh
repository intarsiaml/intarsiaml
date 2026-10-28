#!/usr/bin/env bash
# Lane vrc (ON tatooine, via window.sh STAGES=bench): GEMV / GDN microbench on card 0, plus the bitwise hash tests of the kernels the lane touches.
set -uo pipefail
O=${1:?out dir}
source "$HOME/intarsiaml-vrc/.vrc/lib.sh"
D=dev
clear 1 || exit 2
E="VRC_BENCH=1 ${BENCHE:-}" htest "$D" bench 0 900 --test verify_gemv_bench -- --nocapture
grep -a VRCBENCH "$O/htest-bench.log" | tee -a "$O/bench.txt"
for h in a8_fuse=mxfp4_a8_fuse; do
  t=${h#*=}
  E="${HASHE:-}" htest "$D" "hash-$t" 0 900 --test "$t" -- --nocapture
  grep -aE 'FUSEHASH|VRCHASH' "$O/htest-hash-$t.log" | tee -a "$O/hash.txt"
done
