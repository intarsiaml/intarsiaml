#!/usr/bin/env bash
# Lane mx window 10 (card 0): the head recipe (chunk 4096 + peel 16) on the two 27B paths no earlier window ran: serve
# --max-batch 16 capacity (startup log: context tokens admitted, batching line) at --max-seq 8192 and 17408, main vs head;
# MTP speculative decode (`generate --spec mtp`, lane sp's dense head) 256 greedy ids short / 4k main vs head.
set -uo pipefail
O=$1; S="$(dirname "${BASH_SOURCE[0]}")"; source "$S/lib.sh"
for t in base dev; do [[ $(T $t)/mx-intarsia -nt $(T $t)/Cargo.toml ]] || { log "NO FRESH BINARY in $(T $t); abort"; exit 3; }; done
log "w10 start; budget $(left) s; base $(cat $(T base)/.stamp) dev $(cat $(T dev)/.stamp)"
for ms in 8192 17408; do
  for t in base dev; do
    # timeout runs INSIDE the container, so the server exits with it (no orphan holding the card)
    need 150 "serve" && { INTARSIA_REPO=$(T $t) INTARSIA_TARGET_VOL=$(V $t) INTARSIA_MODELS="$(mdir "$M27")" HIP_VISIBLE_DEVICES=$G timeout 150 "$(T $t)/deploy/tatooine/dev.sh" \
      timeout -s INT 100 /work/mx-intarsia serve --backend hip --model "$M27" --mem-budget 29 --max-seq $ms --max-batch 16 --port 8399 \
      > "$O/serve-$t-mb16-$ms.out" 2> "$O/serve-$t-mb16-$ms.err" < /dev/null
      log "serve $t mb16 max-seq $ms: $(grep -ahE 'prefill chunk|continuous batching|loaded|Error|error' "$O/serve-$t-mb16-$ms.out" "$O/serve-$t-mb16-$ms.err" | cut -c1-200 | tr '\n' '|')"; }
  done
done
for k in short 4k; do
  need 90 "ids" && ids dev mtp-dev-$k "$M27" $k --spec mtp
  need 90 "ids" && ids base mtp-base-$k "$M27" $k --spec mtp
  same ids-mtp-dev-$k.ids ids-mtp-base-$k.ids "27B $k --spec mtp ids head vs main"
  log "spec $k: $(grep -ahE '^spec:' "$O/ids-mtp-dev-$k.err" "$O/ids-mtp-base-$k.err" | tr '\n' '|')"
done
log "w10 done; $(left) s left"
