#!/bin/zsh
# usage: trace.sh <outdir> <preBin> <newBin> <tag...>      (tags: 1k 4k 16k; run inside a fleetq slot)
# Per-call attention GPU time in a real decode: kernel mode (every launch its own command buffer), 129 forced tokens = 128
# decode steps; analysed with b2a/attn-trace.py (files kernel-pre-<tag>.jsonl, kernel-new-<tag>.jsonl, <b>-<tag>.err).
O=$1; PRE=$2; NEW=$3; shift 3
R=/opt/code/intarsiaml/.claude/worktrees/agent-a8c6e3392309dd202/docs/results/raw/2026-10-14-metal-phase-b
M=/Users/josef/models/Qwen3-8B
mkdir -p $O
util() { ioreg -r -c IOAccelerator -d 1 | grep -o '"Device Utilization %"=[0-9]*' | head -1 | cut -d= -f2; }
for P in "$@"; do
  S=8192; [ $P = 16k ] && S=17000
  F=$R/b2a/prompt-$P.txt; [ $P = 4k ] && F=$R/prompt-4k.txt
  for B in pre new; do
    BIN=$PRE; [ $B = new ] && BIN=$NEW
    u=$(util); tries=0; while [ "${u:-99}" -gt 10 ] && [ $tries -lt 20 ]; do sleep 2; u=$(util); tries=$((tries+1)); done
    echo "$B $P idle_util=$u(waits=$tries)"
    INTARSIA_METAL_PROFILE=kernel:$O/kernel-$B-$P.jsonl $BIN generate --model $M --backend metal --max-seq $S --profile-every 0 --temperature 0 --max-tokens 129 --ignore-eos --prompt "$(cat $F)" > $O/$B-$P.out 2> $O/$B-$P.err
  done
done
