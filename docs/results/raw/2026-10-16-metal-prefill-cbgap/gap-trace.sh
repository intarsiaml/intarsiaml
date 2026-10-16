#!/bin/zsh
# Baseline: where is GPU idle between command buffers in a chunked prefill. cb-mode trace, 4062 tokens, chunk 256 and 2048, 2 runs each.
# gap-trace.sh <binary> <outdir>
BIN=$1; O=$2; mkdir -p $O
R=/opt/code/intarsiaml-cbgap/docs/results/raw/2026-10-16-metal-prefill-cbgap
M=/Users/josef/models/Qwen3-8B
export INTARSIA_CPU_FREE_MEM_GB=60
source $R/waitidle.sh
for r in 1 2; do for C in 256 2048; do
  F=$O/cb-4062-c$C-r$r
  wait_idle || touch $F.contaminated
  macmon pipe -i 200 > $F.macmon.jsonl 2>/dev/null & MP=$!
  INTARSIA_METAL_PROFILE=cb:$F.jsonl $BIN generate --model $M --backend metal --max-seq 8192 --temperature 0 --max-tokens 1 \
    --profile-every 0 --prefill-chunk $C --prompt "$(cat $R/prompt-4062.txt)" > $F.out 2> $F.err
  kill $MP 2>/dev/null
done; done
echo done
