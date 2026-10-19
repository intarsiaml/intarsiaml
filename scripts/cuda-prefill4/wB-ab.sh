#!/usr/bin/env bash
# Window B (spark1, vLLM down, INSIDE scripts/cuda-decode/window.sh): speed A/B of the prefill-4 arms, `intarsia bench speed`
# against a fresh `serve` per arm (scripts/cuda-decode/w-ab.sh), sizes 1024 / 4096 / 16384, gen 128, 2 reps per size,
# ROUNDS (3) rounds interleaved: the arm list as given in odd rounds, reversed in even rounds (base,dev / dev,base /
# base,dev for two arms). All arms are one binary (tree p4-dev) unless an arm names another tree.
#   ARMS="base:p4-dev:INTARSIA_CUDA_FP8_GEMM_REG=0 lanes:p4-dev:INTARSIA_CUDA_FP8_GEMM_REG=0,INTARSIA_CUDA_GDN_LANES=1 ..."
# A round is skipped when less than 1.1x the duration of the previous round is left (W_BUDGET_SEC, default 780).
# Output ~/intarsia-decfuse/out/p4B/ab
set -u
P=$HOME/intarsia-decfuse
D=$P/p4-dev/scripts/cuda-decode
export SIZES=${SIZES:-1024,4096,16384} GEN=${GEN:-128} REPS=${REPS:-2} OUT=$P/out/${OUTNAME:-p4B}/ab
DEADLINE=$(( $(date +%s) + ${W_BUDGET_SEC:-780} ))
left() { echo $(( DEADLINE - $(date +%s) )); }
mkdir -p $OUT
read -r -a arms <<< "${ARMS:?arm list}"
gb=$(( $(awk '/MemAvailable/ {print $2}' /proc/meminfo) / 1024 / 1024 )); echo "MemAvailable ${gb} GB"; (( gb >= 90 )) || { echo "ABORT: memory"; exit 1; }
echo "bin: $($P/p4-dev/target/acc/release/intarsia --version 2>&1 | head -1)"
nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv
last=$(( 90 * ${#arms[@]} ))
for r in $(seq 1 ${ROUNDS:-3}); do
  if (( $(left) * 10 < last * 11 )); then echo "SKIP round $r: $(left) s left, the previous round took $last s"; continue; fi
  echo "== round $r ($(left) s left) $(date +%T)"
  t0=$(date +%s)
  if (( r % 2 == 1 )); then order=("${arms[@]}"); else order=(); for ((i=${#arms[@]}-1; i>=0; i--)); do order+=("${arms[$i]}"); done; fi
  bash $D/w-ab.sh 35b $r "${order[@]}"
  last=$(( $(date +%s) - t0 )); echo "round $r took $last s"
done
echo "== p4B done, $(left) s left $(date +%T)"
