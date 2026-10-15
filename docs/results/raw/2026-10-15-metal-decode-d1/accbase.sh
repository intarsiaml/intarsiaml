#!/bin/zsh
R=/opt/code/intarsiaml/.claude/worktrees/agent-a492fe7bd7af6756c/docs/results/raw/2026-10-15-metal-decode-d1
B=/Users/josef/.local/state/claude-watch/d1bin/intarsia-base-766c723
M=/Users/josef/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
mkdir -p $R/gates/accuracy
$B accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --gate-class bf16 --max-seq 4096 > $R/gates/accuracy/base-short.out 2> $R/gates/accuracy/base-short.err
$B accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --gate-class bf16 --max-seq 8192 > $R/gates/accuracy/base-long.out 2> $R/gates/accuracy/base-long.err
