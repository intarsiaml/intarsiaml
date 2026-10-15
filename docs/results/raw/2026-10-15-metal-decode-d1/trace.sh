#!/bin/zsh
# usage: trace.sh <bin> <out-prefix>
# Kernel-mode GPU trace (INTARSIA_METAL_PROFILE=kernel) of a 129-token greedy decode on prompt-short, in one fleetq slot.
BIN=$1; OUT=$2
D=/opt/code/intarsiaml/.claude/worktrees/agent-a492fe7bd7af6756c/docs/results/raw/2026-10-15-metal-decode-d1
fleetq run --host endor --exclusive --mem 24 --label "metal D1: decode op list trace" -- /bin/zsh -c "INTARSIA_METAL_PROFILE=kernel:$OUT.jsonl $BIN generate --model /Users/josef/models/Qwen3-8B --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 129 --ignore-eos --prompt \"\$(cat $D/prompt-short.txt)\" > $OUT.out 2> $OUT.err"
