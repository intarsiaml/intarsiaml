#!/bin/zsh
# Paired, interleaved A/B of `intarsia serve --max-batch 8` arms (speculation on/off, --spec-off-at values) inside ONE
# exclusive fleetq slot (this script takes none itself):
#
#   fleetq run --host endor --exclusive --mem 24 --label "..." -- tools/cbm3_slot.sh MODE OUTDIR [ROUNDS=3]
#
# MODE client    tools/cbm3.py scenarios (env SCEN, default the cbm3.py default set)
# MODE localeval localeval speed --sizes 1024 --gen 512 --reps 3 --concurrency 1,2,4,8 (deepeval-decode-agg: has aggregate_decode_tok_s)
# env: BIN (intarsia binary, required), COOLDOWN (seconds idle before each server start, default 0; thermal throttling was seen), LE_CONC (localeval concurrency levels, default 1,2,4,8), SERVER_ENV (extra VAR=val words for the server, e.g. INTARSIA_SPEC_TRACE=1), ARMS (default "off on"), MODEL, PORT, SCEN, MAXMIN (no new arm after this many minutes, 38)
# Arm names: off | on | offatN (= --spec ngram --spec-off-at N). Round r runs the arms in order, reversed on even rounds, with a
# fresh server per arm. macmon logs the GPU (100 ms) for the whole slot.
set -u
MODE=${1:?usage: cbm3_slot.sh client|localeval OUTDIR [ROUNDS]}; OUT=${2:?outdir}; ROUNDS=${3:-3}
mkdir -p "$OUT"; OUT=${OUT:A}
T=${0:A:h}
BIN=${BIN:?set BIN to the intarsia binary}
MODEL=${MODEL:-$HOME/models/Qwen3-8B}
PORT=${PORT:-8141}
ARMS=(${=ARMS:-off on})
MAXMIN=${MAXMIN:-38}
SCEN=${SCEN:-solo-grounded,solo-open,mixed-once,mixed-loop}
URL=http://127.0.0.1:$PORT
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
lsof -nP -iTCP:$PORT -sTCP:LISTEN -t >/dev/null 2>&1 && { echo "port $PORT in use" >&2; exit 3; }
exec > >(tee -a "$OUT/slot.log") 2>&1
SPID=""; MPID=""
stop_server() { if [ -n "$SPID" ] && kill -0 "$SPID" 2>/dev/null; then kill "$SPID" 2>/dev/null; for i in {1..20}; do kill -0 "$SPID" 2>/dev/null || break; sleep 0.5; done; kill -0 "$SPID" 2>/dev/null && kill -9 "$SPID" 2>/dev/null; fi; SPID=""; }
cleanup() { stop_server; [ -n "$MPID" ] && kill "$MPID" 2>/dev/null; }
trap cleanup EXIT
trap "exit 143" INT TERM
arm_flags() { case $1 in off) echo "--spec off";; on) echo "--spec ngram";; offat*) echo "--spec ngram --spec-off-at ${1#offat}";; *) echo "bad arm $1" >&2; return 1;; esac; }
START=$(date +%s)
echo "slot job $FLEETQ_JOB $(date); mode $MODE; binary $BIN sha256 $(shasum -a 256 "$BIN" | cut -c1-16); arms $ARMS rounds $ROUNDS scen $SCEN"
ps -axo pid,rss,%cpu,etime,command | grep -iE "kev|tensorfold|omlx|splash|llama|vllm|mlx|ollama|intarsia|python" | grep -v "grep\|ps -axo" | cut -c1-170 > "$OUT/ps-start.txt"
macmon pipe -i 100 > "$OUT/macmon.jsonl" 2>/dev/null &
MPID=$!
sleep 1
for r in $(seq 1 $ROUNDS); do
  if (( r % 2 )); then ORDER=($ARMS); else ORDER=(${(Oa)ARMS}); fi
  pos=1
  for c in $ORDER; do
    if (( $(date +%s) - START > MAXMIN * 60 )); then echo "SKIP round $r arm $c (past ${MAXMIN} min)"; continue; fi
    [ "${COOLDOWN:-0}" -gt 0 ] && sleep ${COOLDOWN}
    echo "=== round $r arm $c (position $pos) $(date +%T)"
    FL=$(arm_flags $c) || exit 4
    echo "$(date +%s) round $r arm $c pos $pos" >> "$OUT/arm-times.txt"
    env INTARSIA_CB_TRACE=0 ${=SERVER_ENV:-} "$BIN" serve --model "$MODEL" --backend metal --max-seq 8192 --profile-every 0 --no-supervisor --no-ui \
      --host 127.0.0.1 --port $PORT --max-batch 8 ${=FL} > "$OUT/server-$c-$r.log" 2>&1 &
    SPID=$!
    ok=0
    for i in {1..180}; do
      curl -sf "$URL/health" >/dev/null 2>&1 && { ok=1; break; }
      kill -0 "$SPID" 2>/dev/null || break
      sleep 1
    done
    if [ $ok = 0 ]; then echo "server $c r$r failed to start"; tail -5 "$OUT/server-$c-$r.log"; stop_server; continue; fi
    if [ $MODE = client ]; then
      uv run "$T/cbm3.py" run --url "$URL" --arm $c --round $r --out "$OUT" --scenarios "$SCEN" || echo "client rc=$?"
    else
      ( cd /opt/code/deepeval-decode-agg && uv run localeval speed --url "$URL/v1" --model Qwen3-8B --sizes 1024 --gen 512 --reps 3 \
          --concurrency ${LE_CONC:-1,2,4,8} --label "m3gates-$c-r$r" ) > "$OUT/localeval-$c-$r.log" 2>&1
      echo "localeval rc=$?"
    fi
    curl -s "$URL/metrics" > "$OUT/metrics-$c-$r.txt" 2>/dev/null
    stop_server
    pos=$((pos+1))
  done
done
kill "$MPID" 2>/dev/null; MPID=""
[ $MODE = client ] && { uv run "$T/cbm3.py" analyze --dir "$OUT" --md "$OUT/summary.md" > "$OUT/analyze.log" 2>&1; echo "analyze rc=$?"; }
pgrep -fl "intarsia.* serve" || echo "no intarsia serve left"
echo "done $(date)"
