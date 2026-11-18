#!/usr/bin/env bash
# sjfflip: spark1 gates for the INTARSIA_PREFILL_ORDER default flip (vLLM stays up). Run inside a fleetq slot after build-both.sh:
#   fleetq run --host spark1 --quiet gpu,cpu --mem 52 --expect 40 --label "sjfflip gates" -- scripts/sjfflip/run.sh
# 1. `plan show --json` main vs new, byte-identical (no model weights loaded; order is not in the plan).
# 2. Three servers, `serve --max-batch 8`: main binary (fifo), new binary with the knob unset (shortest), new binary with
#    INTARSIA_PREFILL_ORDER=fifo. Each: scripts/sjf/ident.py (greedy ids, every request solo vs contending: long prompt first, shorts
#    arriving during its prefill; also reports reordered decisions from /metrics), then scripts/sjf/lat.py pair (40K prompt at t=0, 1K
#    prompt at t=2 s: one TTFT line per arm). Then ident.py compare across the three arms.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
S=$REPO/scripts/sjf; OUT=$REPO/docs/results/2026-11-18-sjfflip/raw; mkdir -p "$OUT"
HOST=spark1; HN=spark-4e77; PORT=8232
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
L='~/intarsiaml-sjf'
SPID=""; WPID=""
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/run.log"; }
stop_server() {
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  [[ -n "$SPID" ]] && { rsh "pkill -TERM -P $SPID 2>/dev/null; kill $SPID 2>/dev/null; for i in \$(seq 1 30); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; kill -9 $SPID 2>/dev/null; sleep 3"; SPID=""; }
}
trap stop_server EXIT; trap 'stop_server; exit 130' INT TERM
start_serve() { # start_serve <name> <bin> "<ENV=.. ENV=..>"
  local name=$1 bin=$2 envs=$3
  log "serve $name: $bin env [$envs] serve --backend cuda --max-seq 49152 --mem-budget 40 --max-batch 8"
  rsh "rm -f $L/out/mem-$name.txt"
  SPID=$(rsh "nohup env $envs $L/bin/$bin serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name guidance --max-seq 49152 --mem-budget 40 --max-batch 8 --profile-every 0 > $L/out/serve-$name.log 2>&1 < /dev/null & echo \$!")
  WPID=$(rsh "nohup bash $L/memwatch.sh $SPID $name > /dev/null 2>&1 < /dev/null & echo \$!")
  local i up=0; for i in $(seq 1 240); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
  [[ $up == 1 ]] || { log "serve $name did not come up"; rsh "tail -20 $L/out/serve-$name.log" | tee -a "$OUT/run.log"; stop_server; return 1; }
  log "serve $name up; MemAvailable MB $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo")"
}
finish_serve() {
  curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$1.metrics" 2>/dev/null
  rsh "cat $L/out/serve-$1.log" > "$OUT/serve-$1.log"
  log "min MemAvailable MB ($1): $(rsh "sort -k2 -n $L/out/mem-$1.txt | head -1"), killed: $(rsh "grep -c KILLED $L/out/mem-$1.txt")"
  stop_server
}
scp -q "$S/memwatch.sh" $HOST:$L/memwatch.sh
log "start; earlyoom: $(rsh 'systemctl is-active earlyoom'); vLLM: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:8000/v1/models); main: $(rsh "$L/bin/intarsia-main --version | head -1"); new: $(rsh "$L/bin/intarsia-new --version | head -1")"
for t in main new; do
  rsh "$L/bin/intarsia-$t plan show --backend cuda --model $M35 --max-seq 8192 --mem-budget 30 --json < /dev/null > $L/out/plan-$t.json 2> $L/out/plan-$t.err"
  rsh "cat $L/out/plan-$t.json" > "$OUT/plan-$t.json"
done
if cmp -s "$OUT/plan-main.json" "$OUT/plan-new.json" && [[ -s "$OUT/plan-main.json" ]]; then log "PLAN: plan show --json byte-identical ($(wc -c < "$OUT/plan-main.json") bytes)"; else log "PLAN: DIFFERENT or empty"; fi
URL=http://$HN:$PORT/v1
for arm in "main|intarsia-main|INTARSIA_CB_NO_REUSE=1" "default|intarsia-new|INTARSIA_CB_NO_REUSE=1" "fifo|intarsia-new|INTARSIA_CB_NO_REUSE=1 INTARSIA_PREFILL_ORDER=fifo"; do
  IFS='|' read -r name bin envs <<< "$arm"
  start_serve "$name" "$bin" "$envs" || continue
  python3 "$S/ident.py" run --url "$URL" --metrics "http://$HN:$PORT/metrics" --out "$OUT/ids-$name.json" --repeat 1 2>&1 | tee "$OUT/ident-$name.out" | tail -3
  python3 "$S/lat.py" pair --url "$URL" --arm "$name" | tee -a "$OUT/lat.jsonl"
  finish_serve "$name"
done
python3 "$S/ident.py" compare "$OUT/ids-main.json" "$OUT/ids-default.json" "$OUT/ids-fifo.json" | tee "$OUT/ident-compare.txt"
log "done"
