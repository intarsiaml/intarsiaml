#!/usr/bin/env bash
# O1: the Phase 1 speed gate (docs/ARCHITECTURE.md:322) re-run on current main against same-day vLLM on spark1, by localeval.
# Runs on endor INSIDE one exclusive spark1 slot; vLLM (the fleet `guidance` role) is never touched:
#   fleetq run --host spark1 --exclusive --mem 32 --expect 70 --label "og: ..." -- scripts/og-gates/o1-driver.sh
# Starts `intarsia serve` (tree og-main, DEFAULTS, same serve flags as every earlier localeval arm) on spark1:8090 beside
# the idle vLLM, then ROUNDS rounds of `localeval speed` (sizes SIZES, gen 128, 3 reps, thinking off), vLLM first in odd
# rounds and ours first in even rounds. A 2 s poller records vLLM's running/waiting request counts for the whole slot, so
# a round in which somebody else loaded vLLM while ours was measured can be dropped (o1-analyze.py does that).
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run --host spark1 --exclusive}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT=${OUT:-$REPO/docs/results/raw/2026-10-22-og-gates/o1}; mkdir -p "$OUT"
HOST=spark1; VURL=http://spark-4e77:8000; OURL=http://spark-4e77:8090
ROUNDS=${ROUNDS:-3}; SIZES=${SIZES:-128,1024,4096,16384}
TREE=${TREE:-og-main}
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
BIN="\$HOME/intarsia-decfuse/$TREE/target/acc/release/intarsia"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
log() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$OUT/driver.log"; }
SPID=""; POLL=""
cleanup() {
  trap - EXIT INT TERM; set +e
  [[ -n "$POLL" ]] && kill "$POLL" 2>/dev/null
  if [[ -n "$SPID" ]]; then
    log "stopping our server (pid $SPID on $HOST)"
    rsh "kill -TERM $SPID; for i in \$(seq 1 40); do kill -0 $SPID 2>/dev/null || break; sleep 1; done; kill -KILL $SPID 2>/dev/null; true"
  fi
  rsh "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader" > "$OUT/gpu-after.txt" 2>&1
  log "cleanup done"
}
trap cleanup EXIT INT TERM

log "job $FLEETQ_JOB; tree $TREE; sizes $SIZES; rounds $ROUNDS"
mkdir -p "$OUT"
rsh "mkdir -p \$HOME/og-data; test -x $BIN && ls -la $BIN && $BIN --version | head -1; echo ik_matmul_fp8_gemm_lt symbol count: \$(grep -c ik_matmul_fp8_gemm_lt $BIN)" > "$OUT/binary.txt" 2>&1 \
  || { log "binary missing on $HOST ($TREE): build it first (scripts/cuda-decode/build.sh $TREE)"; cat "$OUT/binary.txt"; exit 3; }
tee -a "$OUT/driver.log" < "$OUT/binary.txt"
gb=$(( $(rsh "awk '/MemAvailable/ {print \$2}' /proc/meminfo") / 1024 / 1024 ))
log "MemAvailable ${gb} GB"
(( gb >= 38 )) || { log "ABORT: less than 38 GB available next to vLLM (spark1 hung once from an OOM beside the 64 GB vLLM)"; exit 4; }
curl -sf -m 5 "$VURL/v1/models" | grep -q '"id"' || { log "ABORT: vLLM does not answer on $VURL"; exit 5; }
rsh "nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu --format=csv" > "$OUT/gpu-before.txt" 2>&1
rsh "docker ps --format '{{.Names}} {{.Image}}' | head -5; head -3 /proc/meminfo" > "$OUT/box-before.txt" 2>&1

# vLLM load poller (whole slot)
( while true; do m=$(curl -s -m 3 "$VURL/metrics"); r=$(echo "$m" | awk '/^vllm:num_requests_running/ {print int($2)}' | tail -1); w=$(echo "$m" | awk '/^vllm:num_requests_waiting/ {print int($2)}' | tail -1); echo "$(date +%s) ${r:-?} ${w:-?}"; sleep 2; done ) > "$OUT/vllm-poll.log" 2>/dev/null &
POLL=$!

log "starting our server on $HOST:8090 (defaults; --max-seq 20480 --mem-budget 30 --profile-every 0, as in the earlier localeval arms)"
SPID=$(rsh "nohup $BIN serve --backend cuda --model $M35 --port 8090 --host 0.0.0.0 --served-model-name intarsia-35b --max-seq 20480 --mem-budget 30 --profile-every 0 > \$HOME/og-data/serve-o1.log 2>&1 < /dev/null & echo \$!") || { log "serve start failed"; exit 6; }
log "server pid $SPID"
up=0; for i in $(seq 1 120); do curl -sf -m 3 "$OURL/health" >/dev/null && { up=1; break; }; rsh "kill -0 $SPID" >/dev/null 2>&1 || break; sleep 3; done
[[ $up == 1 ]] || { log "server did not come up"; rsh "tail -30 \$HOME/og-data/serve-o1.log" | tee -a "$OUT/driver.log"; exit 7; }
log "server up after ~$((i*3)) s"; rsh "head -20 \$HOME/og-data/serve-o1.log" > "$OUT/serve-head.txt" 2>&1

le() { # le <arm> <round>
  local arm=$1 r=$2 t0 t1 rc
  t0=$(date +%s)
  if [[ $arm == vllm ]]; then
    (cd /opt/code/deepeval && uv run localeval speed spark-guidance --thinking off --sizes "$SIZES" --gen 128 --reps 3 --label "og-o1-vllm-r$r" --wait-idle 300) > "$OUT/le-vllm-r$r.out" 2>&1
  else
    (cd /opt/code/deepeval && uv run localeval speed --url "$OURL/v1" --model intarsia-35b --thinking off --sizes "$SIZES" --gen 128 --reps 3 --label "og-o1-ours-r$r" --wait-idle 300) > "$OUT/le-ours-r$r.out" 2>&1
  fi
  rc=$?; t1=$(date +%s)
  printf '%s\t%s\t%s\t%s\t%s\n' "$arm" "$r" "$t0" "$t1" "$rc" >> "$OUT/windows.tsv"
  log "round $r $arm exit $rc $((t1 - t0)) s"
}
# one unrecorded request so the first measured round does not pay lazy init
curl -s -m 120 "$OURL/v1/chat/completions" -H 'content-type: application/json' -d '{"model":"intarsia-35b","messages":[{"role":"user","content":"Say hi."}],"max_tokens":8}' >/dev/null
for r in $(seq 1 "$ROUNDS"); do
  if (( r % 2 )); then le vllm "$r"; le ours "$r"; else le ours "$r"; le vllm "$r"; fi
done
log "all rounds done"
