#!/usr/bin/env bash
# Lane vat (ON tatooine, via window.sh STAGES=unit): the verify attention row-table launch on card 0: byte-equality test, 4 planted
# faults (each must FAIL), the existing row-table / attention tests, then plan show with the knob on and off.
set -uo pipefail
O=${1:?out dir}
source "$HOME/intarsiaml-vat/.vat/lib.sh"
D=dev
clear 1 || exit 2
htest "$D" vat 0 900 --test verify_attn_rows -- --nocapture
grep -aE 'vat verify_attn_rows|test result' "$O/htest-vat.log" | tee -a "$O/hash.txt"
for m in 1 2 3 4; do
  E="INTARSIA_HIP_VAT_MUT=$m" htest "$D" "mut$m" 0 900 --test verify_attn_rows -- --nocapture
  grep -aE 'test result|differing' "$O/htest-mut$m.log" | head -3 | tee -a "$O/mut.txt"
done
htest "$D" row_tables 0 1200 --test row_tables
htest "$D" attention 0 1200 --test attention --test rows_registry --test rows_exact
plan "$D" 0 8b "$M8"
plan "$D" 0,1 27b-tp2 "$M27" $TP2
plan "$D" 0 35b "$M35"
E="INTARSIA_HIP_VERIFY_ATTN_ROWS=0" plan "$D" 0,1 27b-tp2-old "$M27" $TP2
same plan-27b-tp2.out plan-27b-tp2-old.out "27B tp2 plan json, knob on vs off"
log "unit done"
bash "$HOME/intarsiaml-vat/.vat/stages-phase.sh" "$O"
