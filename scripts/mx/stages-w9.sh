#!/usr/bin/env bash
# Lane mx window 9 (card 0): gates on the tree rebased onto main 5e9cc6da (lane mr's row-exact MXFP4 GEMVs, M <= 16):
# the 27B batched decode == solo step at max-batch 16 (lane mr's S4 test) with the head recipe (chunk 4096 + peel 16),
# HIP kernel tests that touch MXFP4 (v2, fold, rows_exact), kernels test --quick (27B, 35B), batch_recurrent.
set -uo pipefail
O=$1; S="$(dirname "${BASH_SOURCE[0]}")"; source "$S/lib.sh"
[[ $(T dev)/mx-intarsia -nt $(T dev)/Cargo.toml ]] || { log "NO FRESH BINARY in $(T dev); abort"; exit 3; }
log "w9 start; budget $(left) s; dev $(cat $(T dev)/.stamp)"
need 900 "mtest27" && mtest27 dev batched27 --test batch_cuda_real real_35b_batched_decode_is_bitwise -- --test-threads=1 --nocapture
need 200 "hip tests" && hiptest dev mx-rows --test mxfp4_v2 --test mxfp4_fold --test rows_exact -- --test-threads=1
need 120 "kq" && kq dev 27 "$M27"
need 120 "kq" && kq dev 35 "$M35"
need 500 "mtest" && mtest dev recurrent --test batch_recurrent -- --test-threads=1 --include-ignored
log "w9 done; $(left) s left"
