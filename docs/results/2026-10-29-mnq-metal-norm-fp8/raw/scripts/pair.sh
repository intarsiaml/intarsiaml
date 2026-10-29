#!/bin/bash
# paired 4k decode: ABAB..., 256 tokens, main vs dev (default) vs dev level 2
[ -n "${FLEETQ_JOB:-}" ] || exit 2
W=/opt/code/intarsiaml/.worktrees/mnq; O=$W/.lane/pair; mkdir -p $O; M=$HOME/models/Qwen3.6-35B-A3B-NVFP4
r() { INTARSIA_METAL_NORM_FP8=$3 $2 generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 --max-tokens 256 --ignore-eos --prompt "$(cat $W/.lane/prompts/p4k-a.txt)" > $O/$1.out 2> $O/$1.err
  echo "$1 $(md5 -q $O/$1.out) $(grep -oE '[0-9.]+ tok/s' $O/$1.err | tail -1)"; }
for i in 1 2 3 4 5 6 7 8 9 10; do
  r main$i $W/.lane/bin/intarsia-main 1
  r dev$i $W/.lane/bin/intarsia-dev 1
  r devoff$i $W/.lane/bin/intarsia-dev 0
done
