#!/usr/bin/env bash
# Lane mx window 2: equivalence test, ceilings, the prefill-chunk question (ids + speed on main), the f8 kernel's GPU test and
# first A/B (card 0).
set -uo pipefail
O=$1; S="$(dirname "${BASH_SOURCE[0]}")"; source "$S/lib.sh"
for t in base dev; do [[ $(T $t)/mx-intarsia -nt $(T $t)/Cargo.toml ]] || { log "NO FRESH BINARY in $(T $t); abort"; exit 3; }; done
log "w2 start; budget $(left) s; base $(stat -c %y $(T base)/mx-intarsia) dev $(stat -c %y $(T dev)/mx-intarsia)"
need 200 "wmma_eq" && hipbin dev wmma_eq.hip wmma-eq 40000 7
need 260 "ceiling" && ceiling dev
need 300 "f8 test" && hiptest dev mxfp4-f8 --test mxfp4_f8 -- --test-threads=1
need 200 "mxbench" && mxbench dev m4096 4096 20
need 120 "mxbench" && mxbench dev m256 256 20
# chunk: does --prefill-chunk change the 27B's bits on HIP (the 35B's do not, docs/results/2026-10-17-hip-prefill.md)?
need 60 "ids" && ids base c256-4k "$M27" 4k
need 60 "ids" && ids base c4096-4k "$M27" 4k --prefill-chunk 4096
same ids-c256-4k.ids ids-c4096-4k.ids "27B 4k ids chunk 256 vs 4096 (base)"
need 80 "ids" && ids base c256-16k "$M27" 16k
need 80 "ids" && ids base c4096-16k "$M27" 16k --prefill-chunk 4096
same ids-c256-16k.ids ids-c4096-16k.ids "27B 16k ids chunk 256 vs 4096 (base)"
# f8 vs main: ids (no-op), default chunk
need 60 "ids" && ids dev f8-short "$M27" short
need 60 "ids" && ids base base-short "$M27" short
same ids-f8-short.ids ids-base-short.ids "27B short ids f8 vs main"
need 60 "ids" && ids dev f8-4k "$M27" 4k
same ids-f8-4k.ids ids-c256-4k.ids "27B 4k ids f8 vs main (chunk 256)"
need 60 "ids" && ids dev f8-c4096-4k "$M27" 4k --prefill-chunk 4096
same ids-f8-c4096-4k.ids ids-c4096-4k.ids "27B 4k ids f8 vs main (chunk 4096)"
need 80 "ids" && ids dev f8-c4096-16k "$M27" 16k --prefill-chunk 4096
same ids-f8-c4096-16k.ids ids-c4096-16k.ids "27B 16k ids f8 vs main (chunk 4096)"
# speed: chunk sweep on main, f8 at the same chunks
for c in 1024 2048 4096; do need 60 "pf" && pf base c$c-4096 "$M27" 4096 --prefill-chunk $c; done
for c in 256 1024 2048 4096; do need 60 "pf" && pf dev f8-c$c-4096 "$M27" 4096 --prefill-chunk $c; done
need 60 "pf" && E="INTARSIA_HIP_MXFP4_F8=0" pf dev off-c4096-4096 "$M27" 4096 --prefill-chunk 4096
for n in 1024 16384; do need 80 "pf" && pf base c4096-$n "$M27" $n --prefill-chunk 4096; need 80 "pf" && pf dev f8-c4096-$n "$M27" $n --prefill-chunk 4096; done
need 240 "prof f8" && prof dev f8-c4096-4096 "$M27" 4096 --prefill-chunk 4096
need 240 "prof base" && prof base c4096-4096 "$M27" 4096 --prefill-chunk 4096
log "w2 done; $(left) s left"
