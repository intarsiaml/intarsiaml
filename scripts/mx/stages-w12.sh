#!/usr/bin/env bash
# Lane mx window 12 (card 0): final gates on the PR head rebased onto main 5241377b (PR #143, lane sp spec follow-ups,
# which touches intarsia-core model.rs / model/batch.rs / recipe.rs like this branch). No-op first: plan show + 256 greedy
# ids, 27B and 35B, short and 4k, no flags or env, head vs main. Then the w9 gates on the rebased build: HIP MXFP4 tests
# (v2, fold, rows_exact), kernels test --quick (27B, 35B), batch_recurrent, the 27B batched decode == solo test.
set -uo pipefail
O=$1; S="$(dirname "${BASH_SOURCE[0]}")"; source "$S/lib.sh"
for t in base dev; do [[ $(T $t)/mx-intarsia -nt $(T $t)/Cargo.toml ]] || { log "NO FRESH BINARY in $(T $t); abort"; exit 3; }; done
log "w12 start; budget $(left) s; base $(cat $(T base)/.stamp) dev $(cat $(T dev)/.stamp)"
for m in 27 35; do
  mp=$M27; [[ $m == 35 ]] && mp=$M35
  need 60 "plan" && plan base base-$m "$mp"
  need 60 "plan" && plan dev dev-$m "$mp"
  same plan-base-$m.out plan-dev-$m.out "plan show $m main vs head"
  for k in short 4k; do
    need 70 "ids" && ids dev dev-$m-$k "$mp" $k
    need 70 "ids" && ids base base-$m-$k "$mp" $k
    same ids-dev-$m-$k.ids ids-base-$m-$k.ids "${m}B $k ids head vs main"
  done
done
need 200 "hip tests" && hiptest dev mx-rows --test mxfp4_v2 --test mxfp4_fold --test rows_exact -- --test-threads=1
need 120 "kq" && kq dev 27 "$M27"
need 120 "kq" && kq dev 35 "$M35"
need 500 "mtest" && mtest dev recurrent --test batch_recurrent -- --test-threads=1 --include-ignored
need 900 "mtest27" && mtest27 dev batched27 --test batch_cuda_real real_35b_batched_decode_is_bitwise -- --test-threads=1 --nocapture
log "w12 done; $(left) s left"
