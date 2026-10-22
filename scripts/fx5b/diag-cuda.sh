#!/usr/bin/env bash
# Lane FX5b: logit-level decomposition of the 35B `continue` deviation on CUDA (spark1), using FX5's binary (same code as #87).
#   fleetq run --host spark1 --mem 45 --expect 30 --label "fx5b 35B reuse-diag long_docs_4k (vLLM up)" -- scripts/fx5b/diag-cuda.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
HOST=${HOST:-spark1}
RAW=${RAW:-$REPO/docs/results/2026-10-22-fx5b-continue-teacher-forced/raw}
mkdir -p "$RAW"
RPATH='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH;'
rsh() { ssh -n -o BatchMode=yes "$HOST" "$RPATH export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
G='$HOME/intarsiaml-golden/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors'
BIN='$HOME/intarsia-decfuse/fx5/target/acc/release/intarsia'
for chunk in ${CHUNKS:-1024}; do
  rsh "cd ~/intarsia-decfuse/fx5 && $BIN accuracy --model $M35 --backend cuda --golden $G --mode reuse --max-seq 8192 --mem-budget 40 --allow-dirty --reuse-measure --reuse-diag --reuse-prompt long_docs_4k --prefill-chunk $chunk > ~/intarsia-decfuse/out/fx5/diag-c$chunk.log 2>&1; echo exit \$? >> ~/intarsia-decfuse/out/fx5/diag-c$chunk.log"
  rsync -a "$HOST:intarsia-decfuse/out/fx5/diag-c$chunk.log" "$RAW/cuda-diag-c$chunk.log"
  grep -a -v '^$' "$RAW/cuda-diag-c$chunk.log" | tail -40
done
