#!/bin/bash
# quick: kernel-mode 4k profile only. usage: pq.sh BIN OUT
BIN=$1; OUT=$2; mkdir -p $OUT; M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; P=/opt/code/intarsiaml/.worktrees/mgd/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
for stem in warm kernel-4k; do
  env INTARSIA_METAL_PROFILE=kernel:$OUT/$stem.jsonl $BIN generate --model $M --backend metal --max-seq 20000 --temperature 0 --max-tokens 2 --ignore-eos --profile-every 0 --prompt "$(cat $P/prompt-4096.txt)" > $OUT/$stem.out 2> $OUT/$stem.err
done
