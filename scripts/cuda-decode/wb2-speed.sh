#!/usr/bin/env bash
# Window B2 of the CUDA norm + GEMV work (spark1, vLLM down, INSIDE scripts/cuda-decode/window.sh): qualify and speed, alone on the GPU.
#   wb2-speed.sh <base tree> <dev tree>
# 1. `intarsia qualify --quick-only` of dev on the 35B
# 2. 35B A/B, interleaved rounds (base-dev, dev-base, base-dev), sizes 128,1024,4096 (decode 128/1024/4096, prefill 1024/4096 from the
#    same runs), gen 128, REPS reps, `--require-graphs` (w-ab.sh)
# Env: ROUNDS (3), REPS (3), W_BUDGET_MIN (14: rounds that do not fit are skipped and said so).
set -u
BASE=${1:?base tree}; DEV=${2:?dev tree}
P=$HOME/intarsia-decfuse
D=$(dirname "$0")
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
GD=$HOME/intarsiaml-golden
O=$P/out/b2; rm -rf $O; mkdir -p $O
BD=$P/$DEV/target/acc/release/intarsia
DEADLINE=$(( $(date +%s) + ${W_BUDGET_MIN:-14} * 60 ))
left() { echo $(( (DEADLINE - $(date +%s)) / 60 )); }
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 90 )) || { echo "ABORT: memory"; exit 1; }
nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv
echo "== heavy-tail norm test, unmutated and norm-order mutant $(date +%T)"
for m in unmutated norm-order; do
  ( INTARSIA_REQUIRE_GPU=1 timeout 200 $P/mutbins/$m --test-threads=1 heavy_tail > $O/heavy-$m.log 2>&1; echo "heavy_tail $m rc $? $(grep -E '^test result' $O/heavy-$m.log)" ) &
done
wait
echo "== qualify $(date +%T)"
$BD qualify --model $M35 --backend cuda --quick-only --gen 128 --reps 5 --no-unim --out $O/qualify-35b --golden-dir $GD --mem-budget 45 > $O/qualify-35b.log 2>&1
echo "qualify exit $?"; tail -40 $O/qualify-35b.log | cut -c1-200
export SIZES=128,1024,4096 REPS=${REPS:-3} OUT=$O/ab
for r in $(seq 1 ${ROUNDS:-3}); do
  if (( $(left) < 2 )); then echo "SKIP A/B round $r: $(left) min left"; continue; fi
  if (( r % 2 == 1 )); then bash $D/w-ab.sh 35b $r base:$BASE dev:$DEV; else bash $D/w-ab.sh 35b $r dev:$DEV base:$BASE; fi
done
echo "== wb2 done $(date +%T), $(left) min left"
