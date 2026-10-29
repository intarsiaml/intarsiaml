#!/usr/bin/env bash
# Lane prb served check on endor (Metal, Qwen3-8B BF16), run under `fleetq run --host endor --quiet gpu,cpu --mem 20`.
# For each engine (single-sequence, --max-batch 4): a cold reference server (INTARSIA_CB_NO_REUSE=1 for the batched one; the
# single-sequence engine's reference is the first run of each prompt, see below), the OLD rule (INTARSIA_PREFIX_REUSE=any) and the
# shipped rule, all on one binary. Usage: served.sh BIN OUTDIR
set -uo pipefail
BIN=${1:?}; OUT=${2:?}; mkdir -p "$OUT"
D=$(cd "$(dirname "$0")" && pwd); PORT=8197; M=$HOME/Models/Qwen3-8B
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
  for mode in noreuse old new; do
    case $mode in noreuse) ENVV="INTARSIA_CB_NO_REUSE=1";; old) ENVV="INTARSIA_PREFIX_REUSE=any";; new) ENVV="X=1";; esac
    # the single-sequence engine has no off switch for reuse: its reference is the batched engine's no-reuse run (same bits)
    [[ $eng == single && $mode == noreuse ]] && continue
    serve "$eng-$mode" ${X[@]+"${X[@]}"} || continue
    python3 "$D/served.py" run "http://127.0.0.1:$PORT/v1" $ORDER "$OUT/$eng-$mode.json" > /dev/null
    stop
  done
done
REF="$OUT/batch4-noreuse.json"
for f in batch4-old batch4-new single-old single-new; do
  echo "== $f vs cold reference (batch4 --no-reuse)"; python3 "$D/served.py" compare "$REF" "$OUT/$f.json" | tee "$OUT/compare-$f.log" | tail -n 25
done
