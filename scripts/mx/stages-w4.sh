#!/usr/bin/env bash
# Lane mx window 4 (card 0): the final v2 tile table and the faster fold pre-pass. v2 bitwise test, fold test, tile sweep
# (small tiles), tails self-golden dumps (compared offline with w3's main dumps), 27B golden gate v2 / fold, no-op ids
# (27B short / 4k / 16k, 35B short / 4k), prefill speed main / v2 / fold at 1k / 4k / 16k, chunk 256 and 4096, profiles.
set -uo pipefail
O=$1; S="$(dirname "${BASH_SOURCE[0]}")"; source "$S/lib.sh"
for t in base dev; do [[ $(T $t)/mx-intarsia -nt $(T $t)/Cargo.toml ]] || { log "NO FRESH BINARY in $(T $t); abort"; exit 3; }; done
log "w4 start; budget $(left) s; base $(cat $(T base)/.stamp) dev $(cat $(T dev)/.stamp)"
need 200 "v2 test" && hiptest dev mxfp4-v2 --test mxfp4_v2 -- --test-threads=1 --nocapture
need 150 "fold test" && hiptest dev mxfp4-fold --test mxfp4_fold -- --test-threads=1 --nocapture
need 120 "mxbench" && mxbench dev m4096 4096 10
need 100 "mxbench" && mxbench dev m1024 1024 10
need 100 "mxbench" && mxbench dev m256 256 10
need 150 "acc" && acc dev dev-c256 256
need 150 "acc" && acc dev dev-c4096 4096
need 60 "accg" && accg dev v2
need 60 "accg" && E="INTARSIA_HIP_MXFP4_FOLD=1" accg dev fold
for k in short 4k 16k; do
  need 70 "ids" && ids dev v2-$k "$M27" $k
  need 70 "ids" && ids base base-$k "$M27" $k
  same ids-v2-$k.ids ids-base-$k.ids "27B $k ids v2 vs main"
done
for k in short 4k; do
  need 60 "ids" && ids dev v2-35-$k "$M35" $k
  need 60 "ids" && ids base base-35-$k "$M35" $k
  same ids-v2-35-$k.ids ids-base-35-$k.ids "35B $k ids v2 vs main"
done
for c in 256 4096; do
  for n in 1024 4096 16384; do
    need 60 "pf" && pf base c$c-$n "$M27" $n --prefill-chunk $c
    need 60 "pf" && pf dev v2-c$c-$n "$M27" $n --prefill-chunk $c
    need 60 "pf" && E="INTARSIA_HIP_MXFP4_FOLD=1" pf dev fold-c$c-$n "$M27" $n --prefill-chunk $c
  done
done
need 120 "prof" && prof dev v2-c256-4096 "$M27" 4096
need 120 "prof" && prof dev v2-c4096-4096 "$M27" 4096 --prefill-chunk 4096
need 120 "prof" && E="INTARSIA_HIP_MXFP4_FOLD=1" prof dev fold-c4096-4096 "$M27" 4096 --prefill-chunk 4096
log "w4 done; $(left) s left"
