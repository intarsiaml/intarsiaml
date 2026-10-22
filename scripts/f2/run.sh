#!/usr/bin/env bash
# Lane F2 (diagnose the opt-in fp8 lt GEMM's chat_reuse_4k `continue` failure), spark1 side, ONE slot:
#   fleetq run --host spark1 --mem 45 --label "F2 fp8 lt reuse sweep (vLLM up)" -- scripts/f2/run.sh
# Builds ~/intarsia-decfuse/f2-dev from this tree (BUILD=0 skips), then runs `accuracy --mode reuse --reuse-sweep` arms.
# ARMS: space separated "tag|chunk|from|stride|n|ENV=V,ENV=V". Output ~/intarsia-decfuse/out/f2, copied to $RAW.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
HOST=${HOST:-spark1}
RAW=${RAW:-$REPO/docs/results/2026-10-21-f2-fp8-diagnose/raw}
mkdir -p "$RAW"
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
if [[ "${BUILD:-1}" == 1 ]]; then
  rsh "mkdir -p $R/f2-dev; test -d $R/f2-dev/target || cp -a $R/g1c-dev/target $R/f2-dev/target"
  "$REPO/scripts/cuda-decode/build.sh" f2-dev 2>&1 | tail -8 | tee "$RAW/00-build.out"
fi
ARMS=${ARMS:-"dev-2048|2048|2100|59|42|INTARSIA_X=0 lt-2048|2048|2100|59|42|INTARSIA_CUDA_FP8_GEMM_LT=1 lt-4096|4096|4100|13|41|INTARSIA_CUDA_FP8_GEMM_LT=1 dev-4096|4096|4100|13|41|INTARSIA_X=0"}
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
L='$HOME/intarsiaml-golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors'
rsh "mkdir -p $R/out/f2; nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu,memory.used --format=csv,noheader; free -g | head -2" | tee "$RAW/01-box.out"
for arm in $ARMS; do
  IFS='|' read -r tag chunk from stride n envs <<< "$arm"
  t0=$(date +%s)
  rsh "cd $R/f2-dev && env ${envs//,/ } timeout 1500 target/acc/release/intarsia accuracy --model $M35 --backend cuda --golden $L --mode reuse --reuse-prompt chat_reuse_4k --max-seq 16384 --mem-budget 40 --prefill-chunk $chunk --allow-dirty --reuse-sweep $n --reuse-sweep-from $from --reuse-sweep-stride $stride > $R/out/f2/sweep-$tag.log 2>&1"
  echo "arm $tag exit $? $(( $(date +%s) - t0 )) s: $(rsh "grep -a '^SUM' $R/out/f2/sweep-$tag.log")" | tee -a "$RAW/02-arms.out"
done
rsync -a "$HOST:intarsia-decfuse/out/f2/" "$RAW/"
