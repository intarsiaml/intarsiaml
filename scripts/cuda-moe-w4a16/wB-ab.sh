#!/usr/bin/env bash
# Window B (spark1, vLLM down, INSIDE scripts/cuda-decode/window.sh): speed A/B, base (w4-base = perf/cuda-moe-nvfp4-mma,
# FP4 kernel off) against dev (w4-dev, the W4A16 kernel on by default), `intarsia bench speed` against a fresh `serve` per arm
# (scripts/cuda-decode/w-ab.sh), sizes 1024 / 4096 / 16384, gen 128, 2 reps per size, 3 rounds interleaved (base,dev /
# dev,base / base,dev). A round is skipped when less than 1.1x the duration of the previous round is left.
# Env: W_BUDGET_SEC (script budget, default 470), ROUNDS (3). Output ~/intarsia-decfuse/out/w4B
set -u
P=$HOME/intarsia-decfuse
D=$P/w4-dev/scripts/cuda-decode
export SIZES=${SIZES:-1024,4096,16384} GEN=${GEN:-128} REPS=${REPS:-2} OUT=$P/out/w4B/ab
DEADLINE=$(( $(date +%s) + ${W_BUDGET_SEC:-470} ))
left() { echo $(( DEADLINE - $(date +%s) )); }
mkdir -p $OUT
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 90 )) || { echo "ABORT: memory"; exit 1; }
echo "base: $($P/w4-base/target/acc/release/intarsia --version 2>&1 | head -1)  dev: $($P/w4-dev/target/acc/release/intarsia --version 2>&1 | head -1)"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv
last=150
for r in $(seq 1 ${ROUNDS:-3}); do
  if (( $(left) * 10 < last * 11 )); then echo "SKIP round $r: $(left) s left, the previous round took $last s"; continue; fi
  echo "== round $r ($(left) s left) $(date +%T)"
  t0=$(date +%s)
  if (( r % 2 == 1 )); then
    bash $D/w-ab.sh 35b $r base:w4-base dev:w4-dev
  else
    bash $D/w-ab.sh 35b $r dev:w4-dev base:w4-base
  fi
  last=$(( $(date +%s) - t0 )); echo "round $r took $last s"
done
echo "== w4B done, $(left) s left $(date +%T)"
