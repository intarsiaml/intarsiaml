#!/usr/bin/env bash
# Lane mx window 1: the 27B prefill profile on one R9700 (card 0), main build (tree base). No code change yet.
set -uo pipefail
O=$1; S="$(dirname "${BASH_SOURCE[0]}")"; source "$S/lib.sh"
[[ -x $(T base)/mx-intarsia ]] || { log "NO BINARY in $(T base); abort"; exit 3; }
log "w1 start; budget $(left) s; binary $(stat -c %y $(T base)/mx-intarsia)"
need 200 "wmma_eq" && hipbin base wmma_eq.hip wmma-eq 40000 7
need 200 "pf warm" && pf base 27-4096-a "$M27" 4096
for n in 1024 4096 16384; do need 150 "pf $n" && pf base 27-$n "$M27" $n; done
need 200 "optab 4096" && optab base 27-4096 "$M27" 4096
need 240 "prof 4096" && prof base 27-4096 "$M27" 4096
need 240 "prof 1024" && prof base 27-1024 "$M27" 1024
need 280 "prof 16384" && prof base 27-16384 "$M27" 16384
need 260 "ceiling" && ceiling base
need 150 "pf 35b 4096" && pf base 35-4096 "$M35" 4096
log "w1 done; $(left) s left"
