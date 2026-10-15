#!/bin/zsh
D=/opt/code/intarsiaml/.claude/worktrees/agent-a492fe7bd7af6756c/docs/results/raw/2026-10-15-metal-decode-d1
B=/Users/josef/.local/state/claude-watch/d1bin
$D/ktrace.sh $B/intarsia-d1c c-all ""
$D/ktrace.sh $B/intarsia-d1c c-glu "rmsnorm_add_matmul_glu_bf16_gemv+rmsnorm_add_matmul_qkv_bf16_gemv"
$D/ktrace.sh $B/intarsia-d1c c-off "all"
