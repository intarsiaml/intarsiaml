#!/bin/zsh
# quiet-window A/B: A = build before the GEMV change + before the MTP default (intarsia-m27), B = MTP-default build with --spec off.
# 3 interleaved rounds: prefill at 4k and decode (plain, 128 tokens) per arm.
O=$1; mkdir -p $O; A=/opt/code/intarsiaml/.worktrees/m27/.scratch/intarsia-m27; B=/opt/code/intarsiaml/.worktrees/m27/.scratch/intarsia-def
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; PR=/opt/code/intarsiaml/.worktrees/m27/docs/results/raw/2026-10-15-metal-longctx-profile/prompts/prompt-4096.txt
P="Write a Rust function that parses a CSV line into fields, handling quoted commas, with unit tests."
for r in 1 2 3; do for arm in a b; do
  if [ $arm = a ]; then BIN=$A; else BIN=$B; fi
  $BIN generate --model $M --backend metal --max-seq 8192 --temperature 0 --max-tokens 2 --ignore-eos --profile-every 0 --spec off --repeat 2 --prompt "$(cat $PR)" > $O/r$r-$arm-pf.out 2> $O/r$r-$arm-pf.err
  $BIN generate --model $M --backend metal --max-seq 4096 --temperature 0 --max-tokens 128 --ignore-eos --profile-every 0 --spec off --chat --no-think -p "$P" > $O/r$r-$arm-dec.out 2> $O/r$r-$arm-dec.err
  echo "r$r $arm: $(grep -h 'finish=' $O/r$r-$arm-pf.err | grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)') | $(grep -h 'finish=' $O/r$r-$arm-dec.err | grep -o 'decode [0-9]* ms ([0-9.]* tok/s)')"
done; done
