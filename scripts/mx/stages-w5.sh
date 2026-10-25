#!/usr/bin/env bash
# Lane mx window 5 (card 0): gates. plan show base vs dev (27B, 35B), kernels test --quick (27B, 35B; default and fold),
# qualify --quick-only (acc-short) 27B (default, fold), no-op ids vs the rebased main (27B short / 4k, 35B short / 4k).
# The 35B qualify runs in w6 (budget).
set -uo pipefail
O=$1; S="$(dirname "${BASH_SOURCE[0]}")"; source "$S/lib.sh"
for t in base dev; do [[ $(T $t)/mx-intarsia -nt $(T $t)/Cargo.toml ]] || { log "NO FRESH BINARY in $(T $t); abort"; exit 3; }; done
log "w5 start; budget $(left) s; base $(cat $(T base)/.stamp) dev $(cat $(T dev)/.stamp)"
for m in 27 35; do
  mp=$M27; [[ $m == 35 ]] && mp=$M35
  need 60 "plan" && plan base base-$m "$mp"
  need 60 "plan" && plan dev dev-$m "$mp"
  same plan-base-$m.out plan-dev-$m.out "plan show $m base vs dev"
  need 60 "plan" && E="INTARSIA_HIP_MXFP4_FOLD=1" plan dev fold-$m "$mp"
  same plan-base-$m.out plan-fold-$m.out "plan show $m base vs dev+fold (differs where the fold is selected)"
done
need 300 "kq" && kq dev 27 "$M27"
need 300 "kq" && E="INTARSIA_HIP_MXFP4_FOLD=1" kq dev 27-fold "$M27"
need 300 "kq" && kq dev 35 "$M35"
need 400 "qual" && qual dev 27 "$M27" --golden $G27
need 400 "qual" && E="INTARSIA_HIP_MXFP4_FOLD=1" qual dev 27-fold "$M27" --golden $G27
for k in short 4k; do
  for m in 27 35; do
    mp=$M27; [[ $m == 35 ]] && mp=$M35
    need 70 "ids" && ids dev dev-$m-$k "$mp" $k
    need 70 "ids" && ids base base-$m-$k "$mp" $k
    same ids-dev-$m-$k.ids ids-base-$m-$k.ids "${m}B $k ids dev vs main"
  done
done
log "w5 done; $(left) s left"
