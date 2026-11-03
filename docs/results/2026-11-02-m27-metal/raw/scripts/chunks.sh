#!/bin/zsh
BIN=/opt/code/intarsiaml/.worktrees/m27/.scratch/intarsia-nax; O=/opt/code/intarsiaml/.worktrees/m27/.scratch/pf
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; P=/opt/code/intarsiaml/.worktrees/m27/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
for ck in 256 512 1024 2048 4096; do
  for c in 4096 16384; do
    $BIN generate --model $M --backend metal --max-seq 20000 --temperature 0 --max-tokens 2 --ignore-eos --profile-every 0 --repeat 2 --prefill-chunk $ck --prompt "$(cat $P/prompt-$c.txt)" > $O/ck$ck-$c.out 2> $O/ck$ck-$c.err
    echo "chunk $ck ctx $c: $(grep -h 'finish=' $O/ck$ck-$c.err | grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)' | tr '\n' ' ')"
  done
done
