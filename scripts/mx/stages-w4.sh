#!/usr/bin/env bash
# Lane mx window 4 (card 0): gates. plan show base vs dev (27B, 35B), kernels test --quick (27B, 35B; default and fold),
# batch_recurrent and batch_cuda_real (+ --ignored) on the hip backend, the HIP crate's GPU tests that touch MXFP4.
set -uo pipefail
O=$1; S="$(dirname "${BASH_SOURCE[0]}")"; source "$S/lib.sh"
for t in base dev; do [[ $(T $t)/mx-intarsia -nt $(T $t)/Cargo.toml ]] || { log "NO FRESH BINARY in $(T $t); abort"; exit 3; }; done
log "w4 start; budget $(left) s; base $(cat $(T base)/.stamp) dev $(cat $(T dev)/.stamp)"
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
need 600 "mtest" && mtest dev batch --test batch_recurrent --test batch_cuda_real -- --test-threads=1 --include-ignored
need 300 "hiptests" && hiptest dev mx-rest --test rows_exact --test quantmoe --test mxfp4_decode -- --test-threads=1
log "w4 done; $(left) s left"
