#!/bin/zsh
# usage: profile.sh <tag> <binary> [commit_every]  -- plain / cb / kernel runs at 128 and 4k context; run inside a fleetq slot
TAG=$1; BIN=$2; CE=$3
W=/opt/code/intarsiaml/.claude/worktrees/agent-a40aebdca23def5e0
R=$W/docs/results/raw/2026-10-14-metal-phase-b
O=$R/b2b/profile
M=/Users/josef/models/Qwen3-8B
mkdir -p $O
if [ -n "$CE" ]; then export INTARSIA_METAL_COMMIT_EVERY=$CE; else unset INTARSIA_METAL_COMMIT_EVERY; fi
for P in short 4k; do
  for MODE in plain cb kernel; do
    if [ $MODE = plain ]; then unset INTARSIA_METAL_PROFILE; else export INTARSIA_METAL_PROFILE=$MODE:$O/$TAG-$MODE-$P.jsonl; fi
    $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 129 --ignore-eos --prompt "$(cat $R/prompt-$P.txt)" > $O/$TAG-$MODE-$P.out 2> $O/$TAG-$MODE-$P.err
  done
done
unset INTARSIA_METAL_PROFILE
gzip -f $O/$TAG-kernel-*.jsonl
echo "profile $TAG done"
