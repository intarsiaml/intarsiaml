#!/usr/bin/env bash
# Lane c8k, endor side: freeze `continue` sequences for CPU oracles (fx5c / lt recipe) with the Metal engine
# (`accuracy --mode reuse --reuse-diag --reuse-dump`, one prompt at a time; any engine's cold greedy stream works, the gate
# substitutes the frozen turn 2 into the real runs). Inside a fleetq slot on endor.
# Usage: metal-dump.sh <outdir> <golden> <prompt:chunk> ...   writes <outdir>/<golden-tag>-<prompt>-c<chunk>/{seq.json,run.log}
#   golden: long | 4k  (golden/qwen36-35b-a3b-nvfp4-vllm-{long,4k}-top20.safetensors)
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
OUT=$1; G=$2; shift 2
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN=${BIN:-/opt/code/target-fx5c/release/intarsia}
GOLD=$REPO/golden/qwen36-35b-a3b-nvfp4-vllm-$G-top20.safetensors
for pc in "$@"; do
  IFS=: read -r pr chunk <<< "$pc"; D=$OUT/$G-$pr-c$chunk; mkdir -p "$D"
  echo "== dump $G $pr chunk=$chunk $(date +%T)"
  "$BIN" accuracy --model $HOME/models/Qwen3.6-35B-A3B-NVFP4 --backend metal --golden "$GOLD" --mode reuse --max-seq 16384 --mem-budget 40 \
    --allow-dirty --no-reuse-oracle --reuse-diag --reuse-prompt $pr --prefill-chunk $chunk --reuse-dump "$D/dump" < /dev/null > "$D/run.log" 2>&1
  echo "exit $?"; cp "$D/dump/seq.json" "$D/seq.json" 2>/dev/null; grep -a "dump:" "$D/run.log" | cut -c1-200; ls -la "$D/seq.json" 2>&1 | cut -c1-120
done
echo "== done $(date +%T)"
