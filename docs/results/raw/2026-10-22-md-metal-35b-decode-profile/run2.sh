#!/bin/zsh
# md lane slot 2: why does served decode (~103 tok/s) trail `generate` decode (~123 tok/s)? Run INSIDE an exclusive endor slot.
# usage: run2.sh <intarsia-binary> <out-dir>
BIN=$1; O=$2; M=/Users/josef/models/Qwen3.6-35B-A3B-NVFP4; PORT=8199
HERE=${0:A:h}; mkdir -p $O; PROMPT=$(cat $HERE/prompt-1k.txt)
gen() { tag=$1; shift
  env "$@" $BIN generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 \
    --max-tokens 129 --ignore-eos --prompt "$PROMPT" > $O/$tag.out 2> $O/$tag.err
  echo "$tag: $(grep -E 'finish=' $O/$tag.err | tail -1)" | tee -a $O/summary2.txt; }
serve() { tag=$1; shift   # extra env as args
  env "$@" $BIN serve --model $M --backend metal --max-seq 8192 --profile-every 0 --host 127.0.0.1 --port $PORT \
    --served-model-name md35 > $O/$tag.server.out 2> $O/$tag.server.err &
  SP=$!
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null && break; sleep 1; done
  $BIN bench speed --url http://127.0.0.1:$PORT --model md35 --sizes 1024 --gen 128 --reps 5 --warmup 1 \
    --label $tag --out-dir $O/$tag.bench > $O/$tag.bench.out 2> $O/$tag.bench.err
  kill $SP; wait $SP 2>/dev/null
  echo "$tag:" | tee -a $O/summary2.txt; grep -E "rep:" $O/$tag.bench.err | tee -a $O/summary2.txt; }
date >> $O/summary2.txt
gen genA X=1
serve serveA X=1
gen genB X=1
serve servecb INTARSIA_METAL_PROFILE=cb:$O/servecb.jsonl
gen genC X=1
serve serveB X=1
gen genD X=1
gzip -f $O/*.jsonl
date >> $O/summary2.txt
