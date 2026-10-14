#!/bin/zsh
# usage: prof.sh <tag> <binary> <outdir>   (run inside a fleetq slot: FLEETQ_JOB=<id>)
# kernel-mode trace (one command buffer per launch) of a 129-token forced decode at ~128 and ~4k context
TAG=$1; BIN=$2; O=$3
R=/opt/code/intarsiaml/.claude/worktrees/agent-a8c6e3392309dd202/docs/results/raw/2026-10-14-metal-phase-b
mkdir -p $O
for P in short 4k; do
  INTARSIA_METAL_PROFILE=kernel:$O/$TAG-kernel-$P.jsonl $BIN generate --model /Users/josef/models/Qwen3-8B --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 129 --ignore-eos --prompt "$(cat $R/prompt-$P.txt)" > $O/$TAG-kernel-$P.out 2> $O/$TAG-kernel-$P.err
done
