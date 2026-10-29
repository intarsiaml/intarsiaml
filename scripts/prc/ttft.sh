#!/usr/bin/env bash
# Lane prc: turn-2 time to first token of a ~1.5k-token chat on endor (Metal, Qwen3-8B BF16), the shipped rule against the #233
# rule (INTARSIA_PREFIX_REUSE=exact), 3 conversations each on one server, `serve --max-batch 4`. Run under
# `fleetq run --host endor --quiet gpu,cpu --mem 20`. Usage: ttft.sh BIN OUTDIR [CHARS]
set -uo pipefail
BIN=${1:?}; OUT=${2:?}; CHARS=${3:-6000}; mkdir -p "$OUT"
D=$(cd "$(dirname "$0")" && pwd); PORT=8199; M=$HOME/Models/Qwen3-8B
SPID=""
stop() { [[ -n "$SPID" ]] && { kill $SPID 2>/dev/null; for i in $(seq 1 20); do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -9 $SPID 2>/dev/null; SPID=""; sleep 2; }; }
trap stop EXIT INT TERM
for mode in exact new; do
  case $mode in exact) ENVV="INTARSIA_PREFIX_REUSE=exact";; new) ENVV="X=1";; esac
  env $ENVV "$BIN" serve --backend metal --model "$M" --port $PORT --served-model-name intarsia --max-seq 4096 --max-batch 4 > "$OUT/server-ttft-$mode.log" 2>&1 < /dev/null &
  SPID=$!
  for i in $(seq 1 120); do curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && break; sleep 2; done
  # a warm-up chat (kernel compilation, first-touch), then the timed conversations
  python3 "$D/chat.py" ttft "http://127.0.0.1:$PORT/v1" 1 600 > /dev/null
  echo "== $mode" | tee "$OUT/ttft-$mode.log"
  python3 "$D/chat.py" ttft "http://127.0.0.1:$PORT/v1" 3 "$CHARS" | tee -a "$OUT/ttft-$mode.log"
  stop
done
