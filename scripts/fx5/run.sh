#!/usr/bin/env bash
# Lane FX5 (hybrid prefix-reuse gate), spark1 side, ONE non-exclusive slot (vLLM stays up):
#   fleetq run --host spark1 --mem 45 --expect 90 --label "fx5 35B reuse logit deviation + qualify acc-long (vLLM up)" -- scripts/fx5/run.sh
# 1. builds ~/intarsia-decfuse/fx5 from this tree (BUILD=0 skips)
# 2. `accuracy --mode reuse --reuse-measure` on the 35B long golden: chunk 1024 twice (qualify's chunk) and 2048 once (the
#    CUDA default); each case prints a REUSE-MEASURE line (nu = top-5 log-prob difference, KL cold||warm)
# 3. `qualify` with only the acc-long stage (+ the stages it needs) to show the gate reports 0 skipped
# Output ~/intarsia-decfuse/out/fx5 on the box, pulled to $RAW after every step.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"; cd "$REPO"
HOST=${HOST:-spark1}
RAW=${RAW:-$REPO/docs/results/2026-10-22-fx5-hybrid-reuse-ci/raw}
mkdir -p "$RAW"
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
pull() { rsync -a "$HOST:intarsia-decfuse/out/fx5/" "$RAW/"; }
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
G='$HOME/intarsiaml-golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors'
BIN="$R/fx5/target/acc/release/intarsia"
OUT="$R/out/fx5"
if [[ "${BUILD:-1}" == 1 ]]; then
  rsh "mkdir -p $R/fx5; test -d $R/fx5/target || cp -a $R/g1c-dev/target $R/fx5/target"
  "$REPO/scripts/cuda-decode/build.sh" fx5 2>&1 | tail -8 | tee "$RAW/00-build.out"
fi
rsh "mkdir -p $OUT; test -s $G && ls -la $G && nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; awk '/MemAvailable/ {print \"MemAvailable GB\", \$2/1048576}' /proc/meminfo" | tee "$RAW/01-env.txt"
common="--model $M35 --backend cuda --golden $G --mode reuse --max-seq 8192 --mem-budget 40 --allow-dirty --reuse-require-all --reuse-measure"
for run in "c1024-run1 1024"; do
  set -- $run
  echo "=== reuse $1 (chunk $2)"
  rsh "cd $R/fx5 && $BIN accuracy $common --prefill-chunk $2 > $OUT/reuse-$1.log 2>&1; echo exit \$? >> $OUT/reuse-$1.log"
  pull
  grep -a -e 'REUSE-MEASURE' -e 'prefix reuse:' -e 'prefix-reuse gate' -e '^exit' "$RAW/reuse-$1.log" | tail -30
done
echo "=== qualify acc-long only"
skips=(); for s in kernels gate acc-short acc-format bench acc-4k envelope bench-long ab tp; do skips+=(--skip $s); done
rsh "cd $R/fx5 && $BIN qualify -m $M35 --backend cuda --golden-long $G --out $OUT/qualify --mem-budget 40 --stage-timeout-min 120 ${skips[*]} > $OUT/qualify.log 2>&1; echo exit \$? >> $OUT/qualify.log"
pull
tail -30 "$RAW/qualify.log"
for run in "c1024-run2 1024" "c2048-run1 2048"; do
  set -- $run
  echo "=== reuse $1 (chunk $2)"
  rsh "cd $R/fx5 && $BIN accuracy $common --prefill-chunk $2 > $OUT/reuse-$1.log 2>&1; echo exit \$? >> $OUT/reuse-$1.log"
  pull
  grep -a -e 'REUSE-MEASURE' -e 'prefix reuse:' -e 'prefix-reuse gate' -e '^exit' "$RAW/reuse-$1.log" | tail -30
done
