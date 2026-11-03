#!/bin/zsh
# usage: pf.sh BIN OUTPREFIX [extra generate args]
BIN=$1; O=$2; shift 2
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; P=/opt/code/intarsiaml/.worktrees/m27/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
for c in 1k 4k; do
  if [ $c = 1k ]; then F=/opt/code/intarsiaml/.worktrees/m27/.scratch/prompt-1k.txt; R=2; else F=$P/prompt-4096.txt; R=${R4:-1}; fi
  $BIN generate --model $M --backend metal --max-seq 20000 --temperature 0 --max-tokens 2 --ignore-eos --profile-every 0 --repeat $R "$@" --prompt "$(cat $F)" > $O-$c.out 2> $O-$c.err
done
