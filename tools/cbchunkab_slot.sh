#!/bin/zsh
# Paired, interleaved A/B of `serve --max-batch 8 --busy-chunk C` for C in {256, 320} (tools/cbchunkab.py scenario).
# Runs inside ONE exclusive fleetq slot (it takes none itself):
#
#   fleetq run --host endor --exclusive --mem 24 --label "..." -- tools/cbchunkab_slot.sh OUTDIR [ROUNDS=8]
#
# Every round starts a fresh server per arm (order alternates: odd rounds 256 then 320, even rounds 320 then 256), runs the
# scenario (warm-up pass + measured pass, no idle gap), stops the server. macmon logs the GPU (100 ms) for the whole slot.
# env: BIN (intarsia binary), MODEL, PORT, ARMS ("256 320")
set -u
OUT=${1:?usage: cbchunkab_slot.sh OUTDIR [ROUNDS]}; ROUNDS=${2:-8}
mkdir -p "$OUT"; OUT=${OUT:A}
T=${0:A:h}
BIN=${BIN:?set BIN to the intarsia binary}
MODEL=${MODEL:-$HOME/models/Qwen3-8B}
PORT=${PORT:-8131}
ARMS=(${=ARMS:-256 320})
URL=http://127.0.0.1:$PORT
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
lsof -nP -iTCP:$PORT -sTCP:LISTEN -t >/dev/null 2>&1 && { echo "port $PORT in use" >&2; exit 3; }
exec > >(tee -a "$OUT/slot.log") 2>&1
SPID=""; MPID=""
stop_server() { if [ -n "$SPID" ] && kill -0 "$SPID" 2>/dev/null; then kill "$SPID" 2>/dev/null; for i in {1..20}; do kill -0 "$SPID" 2>/dev/null || break; sleep 0.5; done; kill -0 "$SPID" 2>/dev/null && kill -9 "$SPID" 2>/dev/null; fi; SPID=""; }
cleanup() { stop_server; [ -n "$MPID" ] && kill "$MPID" 2>/dev/null; }
trap cleanup EXIT INT TERM
echo "slot job $FLEETQ_JOB $(date); binary $BIN sha256 $(shasum -a 256 "$BIN" | cut -c1-16); arms $ARMS rounds $ROUNDS"
macmon pipe -i 100 > "$OUT/macmon.jsonl" 2>/dev/null &
MPID=$!
sleep 1
for r in $(seq 1 $ROUNDS); do
  if (( r % 2 )); then ORDER=($ARMS); else ORDER=(${(Oa)ARMS}); fi
  pos=1
  for c in $ORDER; do
    echo "=== round $r arm $c (position $pos) $(date +%T)"
    INTARSIA_CB_TRACE=1 "$BIN" serve --model "$MODEL" --backend metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui \
      --host 127.0.0.1 --port $PORT --max-batch 8 --busy-chunk $c > "$OUT/server-$c-$r.log" 2>&1 &
    SPID=$!
    ok=0
    for i in {1..180}; do
      curl -sf "$URL/health" >/dev/null 2>&1 && { ok=1; break; }
      kill -0 "$SPID" 2>/dev/null || break
      sleep 1
    done
    if [ $ok = 0 ]; then echo "server $c r$r failed to start"; tail -5 "$OUT/server-$c-$r.log"; stop_server; continue; fi
    uv run "$T/cbchunkab.py" scenario --url "$URL" --arm $c --round $r --order $pos --out "$OUT" || echo "scenario rc=$?"
    curl -s "$URL/metrics" > "$OUT/metrics-$c-$r.txt" 2>/dev/null
    stop_server
    pos=$((pos+1))
  done
done
kill "$MPID" 2>/dev/null; MPID=""
uv run "$T/cbchunkab.py" analyze --dir "$OUT" --md "$OUT/summary.md" > "$OUT/analyze.log" 2>&1; echo "analyze rc=$?"
pgrep -fl "intarsia.* serve" || echo "no intarsia serve left"
echo "done $(date)"
