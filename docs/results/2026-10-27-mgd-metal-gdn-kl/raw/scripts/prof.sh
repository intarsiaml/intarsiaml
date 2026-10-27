#!/bin/bash
# usage: prof.sh BIN OUT  (inside a fleetq slot)
BIN=$1; OUT=$2; mkdir -p $OUT; M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; P=/opt/code/intarsiaml/.worktrees/mgd/docs/results/raw/2026-10-15-metal-longctx-profile/prompts
echo "start $(date -Iseconds) rustc=$(pgrep -x rustc | wc -l) cargo=$(pgrep -x cargo | wc -l)" > $OUT/meta.txt
gen() { local stem=$1 ctx=$2; shift 2
  env "$@" $BIN generate --model $M --backend metal --max-seq 20000 --temperature 0 --max-tokens 2 --ignore-eos --profile-every 0 --prompt "$(cat $P/prompt-$ctx.txt)" > $OUT/$stem.out 2> $OUT/$stem.err
  echo "$stem: $(grep -o 'prefill [0-9]* ms ([0-9.]* tok/s)' $OUT/$stem.err | tail -1)"; }
gen warm 4096 A=1
gen plain-4k 4096 A=1
gen kernel-4k 4096 INTARSIA_METAL_PROFILE=kernel:$OUT/kernel-4k.jsonl
gen kernel-16k 16384 INTARSIA_METAL_PROFILE=kernel:$OUT/kernel-16k.jsonl
echo "end $(date -Iseconds) rustc=$(pgrep -x rustc | wc -l) cargo=$(pgrep -x cargo | wc -l)" >> $OUT/meta.txt
