#!/bin/bash
# Lane M35P: localeval speed (1k/4k/16k prefill + decode) on intarsia Metal, 35B then 8B, inside ONE fleetq slot.
# Usage (inside `fleetq run --host endor --quiet gpu,cpu`): speed.sh BIN OUT_DIR
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
BIN=$1; OUT=$2; mkdir -p "$OUT"; PORT=18435
arm() { # label model_dir served_name extra...
  local label=$1 model=$2 name=$3; shift 3
  echo "##### $label $(date -Iseconds) rustc+cargo before: $(pgrep -x rustc | wc -l)+$(pgrep -x cargo | wc -l)"
  local cmd="$BIN serve --backend metal --model $model --served-model-name $name --max-seq 20000 --profile-every 0 --no-supervisor --no-ui --host 127.0.0.1 --port $PORT $*"
  $cmd > "$OUT/$label-server.out" 2> "$OUT/$label-server.err" &
  local sp=$!
  (while true; do echo "$(date +%T) $(ps -axo pid,%cpu,comm -r | sed -n 2,4p | tr -s " " | tr "\n" ";") rustc=$(pgrep -x rustc | wc -l)"; sleep 4; done) > "$OUT/$label-load.log" 2>&1 &
  local lp=$!
  for i in $(seq 1 240); do curl -sf -m 2 http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $sp 2>/dev/null || break; sleep 1; done
  echo "up after ${i}s"
  (cd /opt/code/deepeval && uv run localeval speed --url http://127.0.0.1:$PORT/v1 --model "$name" --sizes 1024,4096,16384 --gen 256 --reps 3 \
     --thinking off --label "m35p-$label" --recipe "$cmd" > "$OUT/$label-localeval.out" 2> "$OUT/$label-localeval.err"; echo "localeval exit $?")
  echo "rustc+cargo after: $(pgrep -x rustc | wc -l)+$(pgrep -x cargo | wc -l)  $(date -Iseconds)"
  kill $lp 2>/dev/null; kill $sp; wait $sp 2>/dev/null
  for i in $(seq 1 30); do lsof -iTCP:$PORT -sTCP:LISTEN -nP > /dev/null 2>&1 || break; sleep 1; done
}
{ date -Iseconds; echo "job $FLEETQ_JOB"; md5 -q "$BIN"; sw_vers | tr '\n' ' '; top -l 1 -o cpu -n 6 -stats pid,command,cpu | tail -7; } > "$OUT/versions.txt" 2>&1
M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4; M8=$HOME/models/Qwen3-8B
for a in ${ARMS:-35b-b1 8b 35b-mb8}; do
  case $a in
    35b-b1) arm $a $M35 intarsia ;;
    8b) arm $a $M8 intarsia ;;
    35b-mb8) arm $a $M35 intarsia --max-batch 8 ;;
  esac
done
