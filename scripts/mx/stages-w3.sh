#!/usr/bin/env bash
# Lane mx window 3 (card 0): v2 MXFP4 GEMM (bitwise v1) test + tile sweep; chunk 256 vs 4096 self-golden dumps on the tails
# golden (main and v2); no-op ids v2 vs main (27B, 35B); speed v2 vs main.
set -uo pipefail
O=$1; S="$(dirname "${BASH_SOURCE[0]}")"; source "$S/lib.sh"
for t in base dev; do [[ $(T $t)/mx-intarsia -nt $(T $t)/Cargo.toml ]] || { log "NO FRESH BINARY in $(T $t); abort"; exit 3; }; done
log "w3 start; budget $(left) s; base $(cat $(T base)/.stamp) dev $(cat $(T dev)/.stamp)"
need 300 "v2 test" && hiptest dev mxfp4-v2 --test mxfp4_v2 -- --test-threads=1
need 200 "mxbench" && mxbench dev m4096 4096 10
need 120 "mxbench" && mxbench dev m1024 1024 10
need 100 "mxbench" && mxbench dev m256 256 10
# chunk: self-golden prefill dumps (16 prompts: P1 short + synth 260, 264, 4089, 4100, 5686, 16036)
need 150 "acc" && acc base base-c256 256
need 150 "acc" && acc base base-c4096 4096
need 150 "acc" && acc dev dev-c256 256
need 150 "acc" && acc dev dev-c4096 4096
# no-op ids: v2 (default) vs main
need 60 "ids" && ids dev v2-short "$M27" short
need 60 "ids" && ids base base-short "$M27" short
same ids-v2-short.ids ids-base-short.ids "27B short ids v2 vs main"
need 60 "ids" && ids dev v2-4k "$M27" 4k
need 60 "ids" && ids base base-4k "$M27" 4k
same ids-v2-4k.ids ids-base-4k.ids "27B 4k ids v2 vs main"
need 60 "ids" && ids dev v2-35-short "$M35" short
need 60 "ids" && ids base base-35-short "$M35" short
same ids-v2-35-short.ids ids-base-35-short.ids "35B short ids v2 vs main"
need 60 "ids" && ids dev v2-35-4k "$M35" 4k
need 60 "ids" && ids base base-35-4k "$M35" 4k
same ids-v2-35-4k.ids ids-base-35-4k.ids "35B 4k ids v2 vs main"
# speed (prefill only), interleaved
for c in 256 4096; do
  need 60 "pf" && pf base c$c-4096 "$M27" 4096 --prefill-chunk $c
  need 60 "pf" && pf dev v2-c$c-4096 "$M27" 4096 --prefill-chunk $c
done
need 240 "prof v2" && prof dev v2-c4096-4096 "$M27" 4096 --prefill-chunk 4096
log "w3 done; $(left) s left"
