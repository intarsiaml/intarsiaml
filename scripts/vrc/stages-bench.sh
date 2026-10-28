#!/usr/bin/env bash
# Lane vrc (ON tatooine, via window.sh STAGES=bench): GEMV / GDN microbench on card 0 under each knob variant (BENCHV, default
# "def r2 fuse"), plus the bitwise hash test of the W4A8 GEMV under each variant and the planted faults of the wide form.
set -uo pipefail
O=${1:?out dir}
source "$HOME/intarsiaml-vrc/.vrc/lib.sh"
D=dev
clear 1 || exit 2
declare -A V=([def]="" [r2]="INTARSIA_HIP_MXFP4_A8_R=2" [fuse]="INTARSIA_HIP_MXFP4_A8_FUSE=1" [r2fuse]="INTARSIA_HIP_MXFP4_A8_R=2 INTARSIA_HIP_MXFP4_A8_FUSE=1")
for v in ${BENCHV:-def r2 fuse}; do
  E="VRC_BENCH=1 ${V[$v]}" htest "$D" "bench-$v" 0 900 --test verify_gemv_bench -- --nocapture
  grep -a VRCBENCH "$O/htest-bench-$v.log" | sed "s/^/$v /" | tee -a "$O/bench.txt"
done
for v in def r2; do
  E="${V[$v]}" htest "$D" "hash-$v" 0 900 --test mxfp4_a8_fuse -- --nocapture
  echo "$v $(grep -aE 'FUSEHASH' "$O/htest-hash-$v.log")" | tee -a "$O/hash.txt"
  E="A8_FUSE_TEST_DEFAULT=1 ${V[$v]}" htest "$D" "hashd-$v" 0 900 --test mxfp4_a8_fuse -- --nocapture
  echo "$v default-fuse $(grep -aE 'FUSEHASH' "$O/htest-hashd-$v.log")" | tee -a "$O/hash.txt"
done
for m in 1 2 3; do
  E="INTARSIA_HIP_A8_MUT=$m" htest "$D" "mut$m" 0 900 --test mxfp4_a8_fuse -- --nocapture
  grep -aE 'test result|panicked|differs|rel_l2' "$O/htest-mut$m.log" | head -3 | tee -a "$O/mut.txt"
done
for t in mxfp4_a8 rows_exact; do htest "$D" "t-$t" 0 1500 --test "$t"; done
