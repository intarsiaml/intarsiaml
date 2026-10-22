#!/usr/bin/env bash
# Lane `on`: make the three goldens of a model from a vLLM-style reference server, on THIS machine (the Mac has uv; spark1 and the
# tatooine dev image have no numpy and no uv, so `qualify` cannot make a golden there). Each is timed.
#   scripts/onboard/make-goldens.sh <out dir> <file prefix> [base-url] [served-model]
# Run inside a fleetq slot that covers the server (FLEETQ_JOB set). Files: <prefix>-vllm-top20 (short, golden.jsonl),
# <prefix>-vllm-long-top20 (golden.jsonl + golden_long.jsonl), <prefix>-vllm-4k-top20 (golden_4k.jsonl); names that
# qualify finds by model name in --golden-dir. Times go to <out dir>/golden-times.txt.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT=${1:?out dir}; PFX=${2:?file prefix}; BASE=${3:-http://spark-4e77:8000/v1}; MODEL=${4:-guidance}
mkdir -p "$OUT"; : > "$OUT/golden-times.txt"
P=$REPO/tools/prompts
gen() { # gen <suffix> <prompt files...>
  local suf=$1; shift
  local t0; t0=$(date +%s)
  uv run --no-project --quiet --with numpy --with safetensors python "$REPO/tools/golden_vllm.py" --base-url "$BASE" --model "$MODEL" \
    --out "$OUT/$PFX-$suf.safetensors" --prompts "$@" > "$OUT/make-$suf.log" 2>&1
  echo "$suf $(( $(date +%s) - t0 ))s $(ls -l "$OUT/$PFX-$suf.safetensors" | awk '{print $5}') bytes" | tee -a "$OUT/golden-times.txt"
}
gen vllm-top20 "$P/golden.jsonl"
gen vllm-long-top20 "$P/golden.jsonl" "$P/golden_long.jsonl"
gen vllm-4k-top20 "$P/golden_4k.jsonl"
