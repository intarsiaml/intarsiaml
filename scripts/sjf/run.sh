#!/usr/bin/env bash
# sjf: serve-level gates on spark1 (vLLM stays up). Run inside a fleetq slot:
#   fleetq run --host spark1 --expect 45 --mem 52 --quiet gpu,cpu --label "sjf ..." -- scripts/sjf/run.sh ident|lat
#   ident: served greedy ids, solo vs contending, fifo / shortest / shortest with INTARSIA_PACK_PREFILL=0 (scripts/sjf/ident.py)
#   lat:   ABBA fifo vs shortest, `serve --max-batch 4`: pair (40K + 1K at +2 s) x2, burst (4 long + 8 short) x2, decode (4 streams) x3
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
PH=${1:?ident|lat}
REPO="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
S=$REPO/scripts/sjf; OUT=$REPO/docs/results/2026-11-17-sjf/raw; mkdir -p "$OUT"
HOST=spark1; HN=spark-4e77; PORT=8231
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BIN='~/intarsiaml-sjf/bin/intarsia'
SPID=""; WPID=""
rsh() { ssh -n -o BatchMode=yes "$HOST" "$@"; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/run-$PH.log"; }
stop_server() {
  [[ -n "$WPID" ]] && { rsh "kill $WPID 2>/dev/null"; WPID=""; }
  [[ -n "$SPID" ]] && { rsh "pkill -TERM -P $SPID 2>/dev/null; kill $SPID 2>/dev/null; for i in \$(seq 1 30); do kill -0 $SPID 2>/dev/null || exit 0; sleep 1; done; pkill -9 -f 'intarsiaml-sjf/bin/intarsia serve'; sleep 3"; SPID=""; }
}
trap stop_server EXIT; trap 'stop_server; exit 130' INT TERM
start_serve() { # start_serve <name> "<ENV=.. ENV=..>" <serve args...>
  local name=$1 envs=$2; shift 2
  log "serve $name: env [$envs] serve --backend cuda --max-seq 49152 --mem-budget 40 --max-batch 4 $*"
  rsh "rm -f ~/intarsiaml-sjf/out/mem-$name.txt"
  SPID=$(rsh "nohup env $envs $BIN serve --backend cuda --model $M35 --port $PORT --host 0.0.0.0 --served-model-name guidance --max-seq 49152 --mem-budget 40 --max-batch 4 --profile-every 0 $* > ~/intarsiaml-sjf/out/serve-$name.log 2>&1 < /dev/null & echo \$!")
  WPID=$(rsh "nohup bash ~/intarsiaml-sjf/memwatch.sh $SPID $name > /dev/null 2>&1 < /dev/null & echo \$!")
  local i up=0; for i in $(seq 1 240); do curl -sf "http://$HN:$PORT/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" 2>/dev/null || break; sleep 2; done
  [[ $up == 1 ]] || { log "serve $name did not come up"; rsh "tail -20 ~/intarsiaml-sjf/out/serve-$name.log" | tee -a "$OUT/run-$PH.log"; stop_server; return 1; }
  log "serve $name up; MemAvailable MB $(rsh "awk '/MemAvailable/ {print int(\$2/1024)}' /proc/meminfo")"
}
finish_serve() { # finish_serve <name>
  curl -sf "http://$HN:$PORT/metrics" > "$OUT/serve-$1.metrics" 2>/dev/null
  rsh "cat ~/intarsiaml-sjf/out/serve-$1.log" > "$OUT/serve-$1.log"
  log "min MemAvailable MB ($1): $(rsh "sort -k2 -n ~/intarsiaml-sjf/out/mem-$1.txt | head -1"), killed: $(rsh "grep -c KILLED ~/intarsiaml-sjf/out/mem-$1.txt")"
  stop_server
}
scp -q "$S/memwatch.sh" $HOST:~/intarsiaml-sjf/memwatch.sh
log "phase $PH start; earlyoom: $(rsh 'systemctl is-active earlyoom'); vLLM: $(curl -s -o /dev/null -w '%{http_code}' http://$HN:8000/v1/models); bin: $(rsh "$BIN --version | head -1")"
URL=http://$HN:$PORT/v1
if [[ $PH == ident ]]; then
  for arm in "fifo|INTARSIA_CB_NO_REUSE=1" "shortest|INTARSIA_CB_NO_REUSE=1 INTARSIA_PREFILL_ORDER=shortest" "shortest-nopack|INTARSIA_CB_NO_REUSE=1 INTARSIA_PREFILL_ORDER=shortest INTARSIA_PACK_PREFILL=0"; do
    name=${arm%%|*}; envs=${arm#*|}
    start_serve "ident-$name" "$envs" || continue
    python3 "$S/ident.py" run --url "$URL" --metrics "http://$HN:$PORT/metrics" --out "$OUT/ids-$name.json" 2>&1 | tee "$OUT/ident-$name.out" | tail -4
    finish_serve "ident-$name"
  done
  python3 "$S/ident.py" compare "$OUT/ids-fifo.json" "$OUT/ids-shortest.json" "$OUT/ids-shortest-nopack.json" | tee "$OUT/ident-compare.txt"
else
  for arm in fifo shortest shortest fifo; do
    n=$((${n:-0}+1)); name="$arm-$n"
    envs=""; [[ $arm == shortest ]] && envs="INTARSIA_PREFILL_ORDER=shortest"
    start_serve "lat-$name" "$envs" || continue
    for r in 1 2; do python3 "$S/lat.py" pair --url "$URL" --arm "$name" | tee -a "$OUT/lat.jsonl"; sleep 3; done
    for r in 1 2; do python3 "$S/lat.py" burst --url "$URL" --arm "$name" | tee -a "$OUT/lat.jsonl"; sleep 3; done
    for r in 1 2 3; do python3 "$S/lat.py" decode --url "$URL" --arm "$name" | tee -a "$OUT/lat.jsonl"; sleep 3; done
    finish_serve "lat-$name"
  done
fi
log "phase $PH done"
