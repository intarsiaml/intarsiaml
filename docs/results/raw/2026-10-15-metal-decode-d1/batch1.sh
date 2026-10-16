#!/bin/zsh
# usage: batch1.sh : one fleetq slot: baseline kernel trace, GLU fusion unit tests
D=/opt/code/intarsiaml/.claude/worktrees/agent-a492fe7bd7af6756c/docs/results/raw/2026-10-15-metal-decode-d1
B=/Users/josef/.local/state/claude-watch/d1bin
mkdir -p $D/trace $D/tests
INTARSIA_METAL_PROFILE=kernel:$D/trace/base-kernel.jsonl $B/intarsia-base-766c723 generate --model /Users/josef/models/Qwen3-8B --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 129 --ignore-eos --prompt "$(cat $D/prompt-short.txt)" > $D/trace/base-kernel.out 2> $D/trace/base-kernel.err
echo "trace exit $?"
$B/glu-test-1 --test-threads=1 > $D/tests/glu-1.log 2>&1
echo "glu tests exit $?"
