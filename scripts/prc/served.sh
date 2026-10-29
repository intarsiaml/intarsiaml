#!/usr/bin/env bash
# Lane prc served check on endor (Metal, Qwen3-8B BF16), run under `fleetq run --host endor --quiet gpu,cpu --mem 20`.
# For each engine (single-sequence, --max-batch 4): the repeated completions of scripts/prb/served.py and the 2-turn chats of
# scripts/prc/chat.py on a cold reference server (INTARSIA_CB_NO_REUSE=1; the single-sequence engine has no off switch, its
# reference is the batched one's), on the #233 rule (INTARSIA_PREFIX_REUSE=exact), the shipped rule, and the old one
# (INTARSIA_PREFIX_REUSE=any), all on one binary. Usage: served.sh BIN OUTDIR
set -uo pipefail
BIN=${1:?}; OUT=${2:?}; mkdir -p "$OUT"
D=$(cd "$(dirname "$0")" && pwd); PORT=8198; M=$HOME/Models/Qwen3-8B
ORDER=3,4,5,9,13,15,3,4,5,9,13,15,3,3,9,9
SPID=""
stop() { [[ -n "$SPID" ]] && { kill $SPID 2>/dev/null; for i in $(seq 1 20); do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -9 $SPID 2>/dev/null; SPID=""; sleep 2; }; }
trap stop EXIT INT TERM
serve() { # name, extra args..., env via ENVV
  local name=$1; shift
  env $ENVV "$BIN" serve --backend metal --model "$M" --port $PORT --served-model-name intarsia --max-seq 4096 "$@" > "$OUT/server-$name.log" 2>&1 < /dev/null &
  SPID=$!
  for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && return 0; sleep 2; done
  echo "server $name did not come up"; return 1
}
for eng in ${ENGS:-batch4 single}; do
  X=(); [[ $eng == batch4 ]] && X=(--max-batch 4)
  for mode in noreuse exact new old; do
    case $mode in noreuse) ENVV="INTARSIA_CB_NO_REUSE=1";; exact) ENVV="INTARSIA_PREFIX_REUSE=exact";; old) ENVV="INTARSIA_PREFIX_REUSE=any";; new) ENVV="X=1";; esac
    [[ $eng == single && $mode == noreuse ]] && continue
    serve "$eng-$mode" ${X[@]+"${X[@]}"} || continue
    python3 "$D/../prb/served.py" run "http://127.0.0.1:$PORT/v1" $ORDER "$OUT/$eng-$mode-completions.json" > /dev/null
    python3 "$D/chat.py" run "http://127.0.0.1:$PORT/v1" "$OUT/$eng-$mode-chat.json" > "$OUT/$eng-$mode-chat.log"
    stop
  done
done
for f in batch4-exact batch4-new batch4-old single-exact single-new single-old; do
  echo "== $f completions vs cold reference (batch4 --no-reuse)"; python3 "$D/../prb/served.py" compare "$OUT/batch4-noreuse-completions.json" "$OUT/$f-completions.json" | tee "$OUT/compare-$f-completions.log" | tail -n 3
  echo "== $f chats vs cold reference (batch4 --no-reuse)"; python3 "$D/chat.py" compare "$OUT/batch4-noreuse-chat.json" "$OUT/$f-chat.json" | tee "$OUT/compare-$f-chat.log" | tail -n 3
done
