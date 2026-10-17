#!/usr/bin/env bash
# Window B (spark1, vLLM down, INSIDE scripts/cuda-decode/window.sh): speed A/B, base (origin/main build, nv-base) against dev
# with INTARSIA_CUDA_MOE_NVFP4=1 (nv-dev), `intarsia bench speed` against a fresh `serve` per arm (scripts/cuda-decode/w-ab.sh),
# sizes 1024 / 4096 / 16384, gen 128, 2 reps per size, 3 rounds interleaved (base,dev / dev,base / base,dev). A round is skipped
# when less than 1.1x the duration of the previous round is left.
# Env: W_BUDGET_SEC (script budget, default 470), ROUNDS (3). Output ~/intarsia-decfuse/out/wB
set -u
P=$HOME/intarsia-decfuse
D=$P/nv-dev/scripts/cuda-decode
export SIZES=${SIZES:-1024,4096,16384} GEN=${GEN:-128} REPS=${REPS:-2} OUT=$P/out/wB/ab
DEADLINE=$(( $(date +%s) + ${W_BUDGET_SEC:-470} ))
left() { echo $(( DEADLINE - $(date +%s) )); }
mkdir -p $OUT
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 90 )) || { echo "ABORT: memory"; exit 1; }
echo "base: $($P/nv-base/target/acc/release/intarsia --version 2>&1 | head -1)  dev: $($P/nv-dev/target/acc/release/intarsia --version 2>&1 | head -1)"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv
last=150
for r in $(seq 1 ${ROUNDS:-3}); do
  if (( $(left) * 10 < last * 11 )); then echo "SKIP round $r: $(left) s left, the previous round took $last s"; continue; fi
  echo "== round $r ($(left) s left) $(date +%T)"
  t0=$(date +%s)
  if (( r % 2 == 1 )); then
    bash $D/w-ab.sh 35b $r base:nv-base dev:nv-dev:INTARSIA_CUDA_MOE_NVFP4=1
  else
    bash $D/w-ab.sh 35b $r dev:nv-dev:INTARSIA_CUDA_MOE_NVFP4=1 base:nv-base
  fi
  last=$(( $(date +%s) - t0 )); echo "round $r took $last s"
done
echo "== wB done, $(left) s left $(date +%T)"
