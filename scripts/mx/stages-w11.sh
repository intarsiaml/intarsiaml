#!/usr/bin/env bash
# Lane mx window 11 (card 0): final no-op of the PR head (recipe table reverted: the 27B keeps chunk 256) against main
# 5e9cc6da: plan show + 256 greedy ids, 27B and 35B, short and 4k, no flags or env.
set -uo pipefail
O=$1; S="$(dirname "${BASH_SOURCE[0]}")"; source "$S/lib.sh"
for t in base dev; do [[ $(T $t)/mx-intarsia -nt $(T $t)/Cargo.toml ]] || { log "NO FRESH BINARY in $(T $t); abort"; exit 3; }; done
log "w11 start; budget $(left) s; base $(cat $(T base)/.stamp) dev $(cat $(T dev)/.stamp)"
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
log "w11 done; $(left) s left"
