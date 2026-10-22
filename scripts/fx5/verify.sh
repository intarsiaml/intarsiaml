#!/usr/bin/env bash
# Lane FX5, spark1: the reuse gate on the 35B long golden with the recipe's measured reuse_logit_tol / reuse_kl_tol, at chunk 1024
# (qualify's) and 2048 (the CUDA default). Rebuilds first (the recipe is compiled in).
#   fleetq run --host spark1 --mem 45 --expect 40 --label "fx5 verify recipe reuse tol (vLLM up)" -- scripts/fx5/verify.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$S/../.." && pwd)"
HOST=${HOST:-spark1}
RAW=${RAW:-$REPO/docs/results/2026-10-22-fx5-hybrid-reuse-ci/raw}
mkdir -p "$RAW"
R='~/intarsia-decfuse'
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
G='$HOME/intarsiaml-golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors'
BIN="$R/fx5/target/acc/release/intarsia"
OUT="$R/out/fx5v"
"$REPO/scripts/cuda-decode/build.sh" fx5 2>&1 | tail -4 | tee "$RAW/10-verify-build.out"
rsh "mkdir -p $OUT"
for c in 1024 2048; do
  echo "=== recipe tolerance, chunk $c"
  rsh "cd $R/fx5 && $BIN accuracy --model $M35 --backend cuda --golden $G --mode reuse --max-seq 8192 --mem-budget 40 --allow-dirty --reuse-require-all --reuse-measure --prefill-chunk $c > $OUT/reuse-recipe-c$c.log 2>&1; echo exit \$? >> $OUT/reuse-recipe-c$c.log"
  rsync -a "$HOST:intarsia-decfuse/out/fx5v/" "$RAW/"
  grep -a -e 'stream tolerance' -e 'prefix reuse:' -e 'prefix-reuse gate' -e '^exit' "$RAW/reuse-recipe-c$c.log" | cut -c1-400
done
