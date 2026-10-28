#!/usr/bin/env bash
# Lane vrc (ON tatooine, via window.sh STAGES=bench): GEMV / GDN microbench on card 0 under the W4A8 fuse knob variants (BENCHV, default
# "def fuse"), and the W4A8 GEMV hash test under each.
set -uo pipefail
O=${1:?out dir}
source "$HOME/intarsiaml-vrc/.vrc/lib.sh"
D=dev
clear 1 || exit 2
declare -A V=([def]="" [fuse]="INTARSIA_HIP_MXFP4_A8_FUSE=1")
for v in ${BENCHV:-def fuse}; do
  E="VRC_BENCH=1 ${V[$v]}" htest "$D" "bench-$v" 0 900 --test verify_gemv_bench -- --nocapture
  grep -a VRCBENCH "$O/htest-bench-$v.log" | sed "s/^/$v /" | tee -a "$O/bench.txt"
done
for v in def fuse; do
  E="${V[$v]}" htest "$D" "hash-$v" 0 900 --test mxfp4_a8_fuse -- --nocapture
  echo "$v $(grep -aE 'FUSEHASH' "$O/htest-hash-$v.log")" | tee -a "$O/hash.txt"
done
