#!/usr/bin/env bash
# Lane vat (ON tatooine, radiance down; first part of stages-final.sh): the verify attention row-table byte-equality test and its 4 planted
# faults (each must FAIL), card 0.
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
log "mut done"
