#!/usr/bin/env bash
# Lane FX10: `intarsia accuracy --mode reuse` on the Metal 35B (endor), long and short (4k) vLLM goldens, prefill chunk 1024 and 2048.
#   fleetq run --host endor --exclusive --mem 30 --expect 60 --label "fx10 metal 35B reuse" -- scripts/fx10/reuse.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$REPO"
RAW=${RAW:-$REPO/docs/results/2026-10-22-fx10-metal-35b-tests/raw}
mkdir -p "$RAW"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-/opt/code/target-fx10}
[[ "${BUILD:-1}" == 1 ]] && cargo build --release -p intarsia-cli --features metal 2>&1 | tail -3 | tee "$RAW/00-build.out"
BIN="$CARGO_TARGET_DIR/release/intarsia"
G=/opt/code/intarsiaml/golden
for g in "long qwen36-35b-a3b-nvfp4-vllm-long-top20 16384" "short qwen36-35b-a3b-nvfp4-vllm-top20 4096"; do
  set -- $g
  for chunk in 1024 2048; do
    echo "=== reuse $1 golden, chunk $chunk"
    "$BIN" accuracy --model $HOME/models/Qwen3.6-35B-A3B-NVFP4 --backend metal --golden "$G/$2.safetensors" --mode reuse \
      --max-seq "$3" --mem-budget 30 --allow-dirty --prefill-chunk "$chunk" > "$RAW/reuse-$1-c$chunk.log" 2>&1
    echo "exit $?" | tee -a "$RAW/reuse-$1-c$chunk.log"
    grep -a -e 'prefix reuse' -e 'prefix-reuse gate' -e 'reuse' "$RAW/reuse-$1-c$chunk.log" | tail -25
  done
done
