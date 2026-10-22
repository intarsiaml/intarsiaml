#!/usr/bin/env bash
# fx3: one interleaved pass (one visit per arm) on a Spark, run INSIDE one fleetq slot (vLLM `guidance`/`retro` stays UP; nothing is taken down).
#
#   fleetq run --host spark2 --exclusive --mem 32 --expect 40 --label "fx3 8B pass p1" -- \
#       env HOST=spark2 MACHINE=spark-fa00 MODEL=8b PASS=p1 ARMS="intarsia vllm sglang llamacpp llamacpp16" PHASES=speed,conc \
#       scripts/standings-fx3/window.sh
#
# Env: HOST (ssh alias), MACHINE (fleetq/tailscale name used in URLs), MODEL (8b | 35b), PASS (id used in file names), ARMS (order = the
# visit order of this pass), PHASES (speed,conc,quality), OUT (raw dir), SIZES, REPS, CONC, QBENCH, QLIMIT, MINGB.
# Arms: intarsia | vllm (8b: own container from the production image; 35b: the LIVE production server, not restarted) | sglang |
#       llamacpp (-np 1, speed phases) | llamacpp16 (-np 16, conc phase only).
# One visit = one server start + (1 warmup + REPS) requests per size (localeval). Every server is stopped before the next starts.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST=${HOST:?}; MACHINE=${MACHINE:?}; MODEL=${MODEL:?}; PASS=${PASS:?}
ARMS=${ARMS:?}; PHASES=${PHASES:-speed,conc}
OUT=${OUT:-$REPO/docs/results/2026-10-22-fx3-spark-standings/raw/$MODEL}; mkdir -p "$OUT"
SIZES=${SIZES:-128,1024,4096,16384}; REPS=${REPS:-3}; CONC=${CONC:-1,4,16}
QBENCH=${QBENCH:-mmlu,gsm8k}; QLIMIT=${QLIMIT:-25}
MINGB=${MINGB:-30}
TREE=${TREE:-fx3-main}
DE=/opt/code/deepeval
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/window-$PASS.log"; }
has() { [[ ",$PHASES," == *",$1,"* ]]; }

PIDFILE='$HOME/fx3/run/server.pid'
M8=$(rsh 'ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1')
M35=$(rsh 'ls -d $HOME/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1')
if [[ $MODEL == 8b ]]; then GGUF='$HOME/fx3/gguf/Qwen3-8B-BF16.gguf'; else GGUF='$HOME/fx3/gguf/Qwen3.6-35B-A3B-NVFP4.gguf'; fi
LLAMA='$HOME/fx3/llama.cpp/build/bin/llama-server'
BIN="\$HOME/intarsia-decfuse/$TREE/target/acc/release/intarsia"
VLLM_IMG=vllm/vllm-openai:nightly-aarch64
if [[ $MODEL == 8b ]]; then M=$M8; IBUDGET=${IBUDGET:-32}; else M=$M35; IBUDGET=${IBUDGET:-30}; fi
CM="/root/.cache/huggingface/${M#*/.cache/huggingface/}"
PROD_PORT=8000; PROD_NAME=guidance

stop_all() {
  rsh "docker rm -f fx3-sglang fx3-vllm >/dev/null 2>&1; if [ -s $PIDFILE ]; then P=\$(cat $PIDFILE); kill \$P 2>/dev/null; for i in \$(seq 1 40); do kill -0 \$P 2>/dev/null || break; sleep 1; done; kill -9 \$P 2>/dev/null; rm -f $PIDFILE; fi; true" >/dev/null 2>&1
}
POLL=""
cleanup() { [[ -n "$POLL" ]] && kill "$POLL" 2>/dev/null; stop_all; }
trap cleanup EXIT INT TERM
rsh 'mkdir -p $HOME/fx3/run $HOME/fx3/logs' >/dev/null
scp -q "$REPO/scripts/standings-fx3/remote/start-sglang.sh" "$HOST:fx3/start-sglang.sh"

# per-arm: port, served name, start command (remote)
start_arm() {
  case $1 in
    intarsia) PORT=8190; NAME=intarsia
      rsh "nohup $BIN serve --backend cuda --model $M --host 0.0.0.0 --port $PORT --served-model-name $NAME --max-seq 20480 --mem-budget $IBUDGET --profile-every 0 --no-supervisor --no-ui > \$HOME/fx3/logs/intarsia-$MODEL-$PASS.log 2>&1 < /dev/null & echo \$! > $PIDFILE";;
    vllm)
      if [[ $MODEL == 35b ]]; then PORT=$PROD_PORT; NAME=$PROD_NAME; return 0; fi   # production server, live
      PORT=8300; NAME=vllm
      rsh "docker run -d --name fx3-vllm --gpus all --ipc=host --network host -v \$HOME/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 -e VLLM_USE_FLASHINFER_MOE_FP4=0 -e VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel $VLLM_IMG $CM --host 0.0.0.0 --port $PORT --served-model-name $NAME --max-model-len 20480 --max-num-seqs 16 --max-num-batched-tokens 8192 --gpu-memory-utilization ${VLLM_UTIL:-0.20} --enable-prefix-caching --enable-chunked-prefill > /dev/null";;
    sglang) PORT=8200; NAME=sglang
      rsh "bash \$HOME/fx3/start-sglang.sh $MODEL $PORT" >/dev/null;;
    llamacpp) PORT=8400; NAME=llamacpp
      rsh "nohup $LLAMA -m $GGUF -ngl 999 -fa on -c 20480 -np 1 --host 0.0.0.0 --port $PORT -a $NAME --jinja --cache-ram 0 --no-webui > \$HOME/fx3/logs/llamacpp-$PASS.log 2>&1 < /dev/null & echo \$! > $PIDFILE";;
    llamacpp16) PORT=8400; NAME=llamacpp
      rsh "nohup $LLAMA -m $GGUF -ngl 999 -fa on -c 81920 -np 16 --host 0.0.0.0 --port $PORT -a $NAME --jinja --cache-ram 0 --no-webui > \$HOME/fx3/logs/llamacpp16-$PASS.log 2>&1 < /dev/null & echo \$! > $PIDFILE";;
    *) log "unknown arm $1"; return 1;;
  esac
}
alive() {
  case $ARM in
    vllm) [[ $MODEL == 35b ]] && return 0; rsh "docker ps -q --filter name=fx3-vllm | grep -q ." ;;
    sglang) rsh "docker ps -q --filter name=fx3-sglang | grep -q ." ;;
    *) rsh "kill -0 \$(cat $PIDFILE) 2>/dev/null" ;;
  esac
}
wait_ready() {  # url timeout_s
  local t=0
  while (( t < $2 )); do
    if (( t >= 15 )) && ! alive; then log "  $1 process/container died while starting"; return 1; fi
    curl -sf -m 5 "http://$MACHINE:$PORT/v1/models" >/dev/null 2>&1 && { log "  $1 ready after ${t}s"; return 0; }
    sleep 5; t=$((t+5))
  done
  return 1
}
loc() {  # localeval subcommand label extra args...
  local sub=$1 label=$2; shift 2
  ( cd $DE && uv run localeval "$sub" --url "http://$MACHINE:$PORT/v1" --model "$NAME" --thinking off --label "$label" "$@" )
}
save_run() {  # arm tag
  local d; d=$(ls -td $DE/results/*"$2"* 2>/dev/null | head -1)
  if [[ -n "$d" ]]; then cp "$d/result.json" "$OUT/$1-$PASS-$3.result.json" 2>/dev/null; cp "$d/summary.md" "$OUT/$1-$PASS-$3.summary.md" 2>/dev/null; fi
}

# Poll the production vLLM (running/waiting) for foreign load during the whole pass.
( while true; do m=$(curl -s -m 3 "http://$MACHINE:$PROD_PORT/metrics" 2>/dev/null); r=$(echo "$m" | awk '/^vllm:num_requests_running/ {print int($2)}' | head -1); w=$(echo "$m" | awk '/^vllm:num_requests_waiting\{/ {print int($2)}' | head -1); echo "$(date +%T) running=${r:-?} waiting=${w:-?}"; sleep 2; done ) > "$OUT/prod-poll-$PASS.log" 2>/dev/null &
POLL=$!

log "pass $PASS model=$MODEL host=$HOST arms=[$ARMS] phases=$PHASES job=$FLEETQ_JOB"
rsh 'nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv; docker ps --format "{{.Names}} {{.Image}}"; grep -E "MemAvailable" /proc/meminfo' > "$OUT/box-before-$PASS.txt" 2>&1
rsh "$BIN --version 2>&1 | head -2" > "$OUT/binary-$PASS.txt" 2>&1

for ARM in $ARMS; do
  stop_all
  sleep 3
  avail=$(rsh "awk '/MemAvailable/ {print int(\$2/1024/1024)}' /proc/meminfo"); log "== arm $ARM (MemAvailable ${avail} GB)"
  if [[ $ARM != vllm || $MODEL == 8b ]] && (( avail < MINGB )); then log "SKIP $ARM: MemAvailable $avail GB < $MINGB"; continue; fi
  start_arm $ARM || continue
  if ! wait_ready $ARM "${READY_TIMEOUT:-1200}"; then
    log "FAIL $ARM did not come up"; rsh "tail -30 \$HOME/fx3/logs/*$PASS.log 2>/dev/null; docker logs fx3-sglang 2>&1 | tail -30; docker logs fx3-vllm 2>&1 | tail -30" > "$OUT/$ARM-$PASS-startfail.txt" 2>&1; stop_all; continue
  fi
  rsh 'nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader' > "$OUT/$ARM-$PASS-gpu.txt" 2>&1
  if has speed && [[ $ARM != llamacpp16 ]]; then
    L="fx3-$MODEL-$ARM-$PASS-speed"
    loc speed "$L" --sizes "$SIZES" --gen 128 --reps "$REPS" > "$OUT/$ARM-$PASS-speed.out" 2>&1; log "  speed exit=$?"; save_run "$ARM" "$L" speed
  fi
  if has conc && [[ $ARM != llamacpp ]]; then
    L="fx3-$MODEL-$ARM-$PASS-conc"
    loc speed "$L" --sizes 1024 --gen 128 --reps 1 --concurrency "$CONC" > "$OUT/$ARM-$PASS-conc.out" 2>&1; log "  conc exit=$?"; save_run "$ARM" "$L" conc
  fi
  if has quality && [[ $ARM != llamacpp16 ]]; then
    L="fx3-$MODEL-$ARM-$PASS-quality"
    loc quality "$L" --structured none --bench "$QBENCH" --limit "$QLIMIT" > "$OUT/$ARM-$PASS-quality.out" 2>&1; log "  quality exit=$?"; save_run "$ARM" "$L" quality
  fi
  rsh 'nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv,noheader' >> "$OUT/$ARM-$PASS-gpu.txt" 2>&1
  stop_all
done
log "pass $PASS done"
