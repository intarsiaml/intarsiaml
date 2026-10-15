#!/bin/zsh
# usage: ktrace.sh <bin> <tag> [D1_OFF list]   (run inside a fleetq slot; kernel-mode trace of a 129-token decode on prompt-short)
BIN=$1; TAG=$2; OFF=$3
D=/opt/code/intarsiaml/.claude/worktrees/agent-a492fe7bd7af6756c/docs/results/raw/2026-10-15-metal-decode-d1
mkdir -p $D/trace
INTARSIA_METAL_D1_OFF=$OFF INTARSIA_METAL_PROFILE=kernel:$D/trace/$TAG-kernel.jsonl $BIN generate --model /Users/josef/models/Qwen3-8B --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 129 --ignore-eos --prompt "$(cat $D/prompt-short.txt)" > $D/trace/$TAG-kernel.out 2> $D/trace/$TAG-kernel.err
python3 $D/trace_sum.py $D/trace/$TAG-kernel.jsonl > $D/trace/$TAG-kernel.sum
echo "== $TAG"; sed -n 1p $D/trace/$TAG-kernel.sum; tail -n +5 $D/trace/$TAG-kernel.sum | head -8
