#!/bin/zsh
# md lane: Qwen3.6-35B-A3B NVFP4 Metal decode profile. Run INSIDE an exclusive endor fleetq slot.
# usage: run.sh <intarsia-binary> <out-dir>
BIN=$1; O=$2; M=/Users/josef/models/Qwen3.6-35B-A3B-NVFP4
HERE=${0:A:h}
mkdir -p $O; PROMPT=$(cat $HERE/prompt-1k.txt)
run() { # tag, then env assignments via `env`
  tag=$1; shift
  : > $O/$tag.mac
  macmon pipe -i 250 > $O/$tag.mac 2>/dev/null &
  MP=$!
  env "$@" $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 \
    --max-tokens 129 --ignore-eos --prompt "$PROMPT" > $O/$tag.out 2> $O/$tag.err
  kill $MP 2>/dev/null
  grep -E "prompt:|prefill" $O/$tag.err | tail -2 | tee -a $O/summary.txt
}
date >> $O/summary.txt
run plain1 X=1
run plain2 X=1
run plain3 X=1
run cb1 INTARSIA_METAL_PROFILE=cb:$O/cb1.jsonl
run cb2 INTARSIA_METAL_PROFILE=cb:$O/cb2.jsonl
run kernel1 INTARSIA_METAL_PROFILE=kernel:$O/kernel1.jsonl
run ce1 INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:$O/ce1.jsonl
run plain4 X=1
run plain5 X=1
gzip -f $O/*.jsonl
date >> $O/summary.txt
