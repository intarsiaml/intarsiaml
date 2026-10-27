#!/bin/bash
# Lane M35P: intarsia accuracy, Qwen3.6-35B-A3B NVFP4 on Metal, against the vLLM goldens (short, long, 4k), inside one fleetq slot.
# Usage: accuracy.sh BIN OUT_DIR
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
BIN=$1; OUT=$2; mkdir -p "$OUT"
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; G=/opt/code/intarsiaml/golden
echo "job $FLEETQ_JOB $(date -Iseconds) rustc=$(pgrep -x rustc | wc -l) cargo=$(pgrep -x cargo | wc -l)" > "$OUT/meta.txt"
run() { # name golden maxseq decode extra...
  local n=$1 g=$2 ms=$3 dp=$4; shift 4
  echo "== $n $(date -Iseconds)"
  $BIN accuracy --model $M --backend metal --golden $G/$g --mode both --decode-positions $dp --max-seq $ms \
     --write-baseline "$OUT/$n.baseline.json" --write-verdict "$OUT/$n.verdict.json" --write-self-golden "$OUT/$n.selfgolden.safetensors" "$@" > "$OUT/$n.log" 2>&1
  echo "exit $? $n"
}
run short qwen36-35b-a3b-nvfp4-vllm-top20.safetensors 4096 16
run long qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors 16384 16
run 4k qwen36-35b-a3b-nvfp4-vllm-4k-top20.safetensors 8192 64
echo "done $(date -Iseconds) rustc=$(pgrep -x rustc | wc -l) cargo=$(pgrep -x cargo | wc -l)"
