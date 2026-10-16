#!/bin/zsh
# Start `intarsia serve` (Qwen3-8B BF16, Metal) and run tools/cbsoak.py against it, memory sampled from the server PID.
# Takes no fleetq slot itself: run it as the command of an exclusive slot, which also covers the GPU use of the server:
#
#   fleetq run --host endor --exclusive --mem 24 --label "cbsoak ..." -- tools/cbsoak_serve.sh OUTDIR --rate 0.5 --duration 180
#
# tools/cbsoak_serve.sh OUTDIR [cbsoak run args...] [--then cbsoak run args...]...   (OUTDIR is created; server.log, soak/, soak2/ ... go there;
#   several runs share one server start, one slot)
# env: INTARSIA_BIN (default: the primary worktree's release binary), MODEL (~/models/Qwen3-8B), PORT (8107),
#      SERVE_EXTRA (extra serve flags, e.g. "--max-batch 4"; word-split).
# Only the server PID this script started is ever signalled.
set -u
OUT=${1:?usage: cbsoak_serve.sh OUTDIR [cbsoak run args...]}; shift; mkdir -p "$OUT"; OUT=${OUT:A}
T=${0:A:h}
BIN=${INTARSIA_BIN:-/opt/code/intarsiaml/target/release/intarsia}
MODEL=${MODEL:-$HOME/models/Qwen3-8B}
PORT=${PORT:-8107}
URL=http://127.0.0.1:$PORT
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
[ -d "$DEVELOPER_DIR" ] || { echo "DEVELOPER_DIR missing" >&2; exit 3; }
[ -x "$BIN" ] || { echo "no binary $BIN" >&2; exit 3; }
[ -n "${FLEETQ_JOB:-}" ] || echo "WARNING: FLEETQ_JOB not set: not inside a fleetq slot" >&2
lsof -nP -iTCP:$PORT -sTCP:LISTEN -t >/dev/null 2>&1 && { echo "port $PORT already in use" >&2; exit 3; }
exec > >(tee -a "$OUT/cbsoak_serve.log") 2>&1
SPID=""; MPID=""
hyg() {  # $1 = label: other GPU/CPU users (ps top 12 by cpu), swap, memory pressure
  { echo "--- $1 $(date '+%H:%M:%S')"; ps -axo pid,pcpu,rss,etime,comm | sort -k2 -nr | head -12 | cut -c1-150; sysctl vm.swapusage; echo "tracked ps: $(pgrep -fl 'kev|tensorfold' | head -5 | tr '\n' ';')"; } >> "$OUT/hygiene.txt" 2>&1
}
cleanup() { [ -n "$MPID" ] && kill "$MPID" 2>/dev/null; hyg "end (cleanup)"; if [ -n "$SPID" ] && kill -0 "$SPID" 2>/dev/null; then kill "$SPID" 2>/dev/null; sleep 3; kill -0 "$SPID" 2>/dev/null && kill -9 "$SPID" 2>/dev/null; fi; }
trap cleanup EXIT INT TERM
hyg "start (before macmon/server)"
if command -v macmon >/dev/null 2>&1; then macmon pipe -i 1000 >"$OUT/macmon.jsonl" 2>/dev/null & MPID=$!; sleep 6; echo "macmon pid $MPID, idle sample (gpu_usage, 6 s before server start):"; tail -n 3 "$OUT/macmon.jsonl" | python3 -c "import sys,json
for l in sys.stdin:
    d=json.loads(l); print('  gpu_usage',d['gpu_usage'],'gpu_power',round(d['gpu_power'],2),'sys_power',round(d['sys_power'],1))"; fi
echo "binary: $BIN  model: $MODEL  port: $PORT  extra: ${SERVE_EXTRA:-}"
"$BIN" serve --model "$MODEL" --backend metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT ${=SERVE_EXTRA:-} >"$OUT/server.log" 2>&1 &
SPID=$!
for i in {1..120}; do
  curl -sf "$URL/health" >/dev/null 2>&1 && break
  kill -0 "$SPID" 2>/dev/null || { echo "server died:"; tail -20 "$OUT/server.log"; exit 6; }
  sleep 2
done
curl -sf "$URL/health" >/dev/null 2>&1 || { echo "server not healthy after 240 s"; tail -20 "$OUT/server.log"; exit 7; }
echo "server up pid=$SPID after ~$((i*2)) s; plan: $(curl -s $URL/v1/intarsia/plan | head -c 600)"
RC=0; N=1; RUN=()
run_one() { local sub=run; case "$1" in scenario|burst|run) sub=$1; shift;; esac; echo "=== run $N ($sub): $* ==="; uv run "$T/cbsoak.py" $sub --url "$URL" --pid "$SPID" --out "$OUT/soak$([ $N -gt 1 ] && echo $N)" "$@"; local r=$?; echo "cbsoak run $N rc=$r"; [ $r -gt $RC ] && RC=$r; N=$((N+1)); }
for a in "$@"; do
  if [ "$a" = "--then" ]; then run_one "${RUN[@]}"; RUN=(); else RUN+=("$a"); fi
done
run_one "${RUN[@]}"
echo "cbsoak rc=$RC"
curl -s "$URL/metrics" > "$OUT/metrics-end.txt" 2>/dev/null
hyg "end (before server stop)"
exit $RC
