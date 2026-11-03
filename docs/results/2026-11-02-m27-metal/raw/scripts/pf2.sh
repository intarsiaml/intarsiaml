#!/bin/zsh
# prefill A/B: usage pf2.sh OUTDIR BIN_A BIN_B ; 3 interleaved rounds, 4k and 16k prompt, plain generate --repeat 2
O=$1; A=$2; B=$3; mkdir -p $O
M=/Users/josef/models/Qwen3.8-27B-Quark-AWQ-MXFP4; P=/opt/code/intarsiaml/.worktrees/m27/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
for r in 1 2 3; do for arm in a b; do
  if [ $arm = a ]; then BIN=$A; else BIN=$B; fi
  for c in 4096 16384; do
    $BIN generate --model $M --backend metal --max-seq 20000 --temperature 0 --max-tokens 2 --ignore-eos --profile-every 0 --repeat 2 --prompt "$(cat $P/prompt-$c.txt)" > $O/r$r-$arm-$c.out 2> $O/r$r-$arm-$c.err
    echo "r$r $arm $c: $(grep -h 'finish=' $O/r$r-$arm-$c.err | grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)')"
  done
done; done
