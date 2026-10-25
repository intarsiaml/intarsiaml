#!/usr/bin/env bash
# Lane mx window 5 (card 0): batch_recurrent and batch_cuda_real (+ ignored) on the hip backend, the HIP crate's GPU tests
# that touch MXFP4 or the registry.
set -uo pipefail
O=$1; S="$(dirname "${BASH_SOURCE[0]}")"; source "$S/lib.sh"
log "w5 start; budget $(left) s; dev $(cat $(T dev)/.stamp)"
need 600 "mtest" && mtest dev batch --test batch_recurrent --test batch_cuda_real -- --test-threads=1 --include-ignored
need 300 "hiptests" && hiptest dev mx-rest --test rows_exact --test quantmoe --test mxfp4_decode -- --test-threads=1
log "w5 done; $(left) s left"
