#!/usr/bin/env bash
# Lane vrc (ON tatooine, via window.sh STAGES=gdn): the GDN verify snapshot launch on card 0: bitwise test, 4 planted faults (each must FAIL),
# the existing fusion / row-table tests of the kernel, and the timing bench.
set -uo pipefail
O=${1:?out dir}
source "$HOME/intarsiaml-vrc/.vrc/lib.sh"
D=dev
clear 1 || exit 2
htest "$D" snap 0 900 --test gdn_verify_snap -- --nocapture
grep -aE 'VRCHASH|VRCMUT' "$O/htest-snap.log" | tee -a "$O/hash.txt"
for m in 1 2 3 4; do
  E="INTARSIA_HIP_GDN_SNAP_MUT=$m" htest "$D" "mut$m" 0 900 --test gdn_verify_snap -- --nocapture
  grep -aE 'test result|differs' "$O/htest-mut$m.log" | head -3 | tee -a "$O/mut.txt"
done
htest "$D" fusion_p2 0 1200 --test fusion_p2
htest "$D" row_tables 0 1200 --test row_tables
E="VRC_BENCH=1" htest "$D" bench 0 900 --test verify_gemv_bench -- --nocapture
grep -a VRCBENCH "$O/htest-bench.log" | tee -a "$O/bench.txt"
