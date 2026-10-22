#!/bin/zsh
# md lane slot 3: is main's ~123 tok/s a code change or machine state? Interleaved A/B of base d99cfa73 (sp-gemv's "103 tok/s" base) vs main.
# usage: run3.sh <main-binary> <base-binary> <out-dir>
NEW=$1; OLD=$2; O=$3; M=/Users/josef/models/Qwen3.6-35B-A3B-NVFP4; PORT=8199
HERE=${0:A:h}; mkdir -p $O; PROMPT=$(cat $HERE/prompt-1k.txt)
gen() { tag=$1; bin=$2; shift 2
  : > $O/$tag.mac; macmon pipe -i 250 > $O/$tag.mac 2>/dev/null & MP=$!
  env "$@" $bin generate --model $M --backend metal --max-seq 8192 --profile-every 0 --temperature 0 \
    --max-tokens 129 --ignore-eos --prompt "$PROMPT" > $O/$tag.out 2> $O/$tag.err
  kill $MP 2>/dev/null
  echo "$tag: $(grep -E 'finish=' $O/$tag.err | tail -1)" | tee -a $O/summary3.txt; }
serve() { tag=$1; bin=$2; shift 2
  env "$@" $bin serve --model $M --backend metal --max-seq 8192 --profile-every 0 --host 127.0.0.1 --port $PORT \
    --served-model-name md35 > $O/$tag.server.out 2> $O/$tag.server.err &
  SP=$!
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:$PORT/v1/models > /dev/null && break; sleep 1; done
  $bin bench speed --url http://127.0.0.1:$PORT --model md35 --sizes 1024 --gen 128 --reps 5 --warmup 1 \
    --label $tag --out-dir $O/$tag.bench > $O/$tag.bench.out 2> $O/$tag.bench.err
  kill $SP; wait $SP 2>/dev/null
  echo "$tag:" | tee -a $O/summary3.txt; grep -E "rep:" $O/$tag.bench.err | tee -a $O/summary3.txt; }
date >> $O/summary3.txt
for i in 1 2 3 4; do gen baseG$i $OLD X=1; gen mainG$i $NEW X=1; done
serve baseS $OLD X=1
serve mainS $NEW X=1
gen baseE $OLD INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:$O/baseE.jsonl
gen mainE $NEW INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:$O/mainE.jsonl
for i in 5 6; do gen baseG$i $OLD X=1; gen mainG$i $NEW X=1; done
gzip -f $O/*.jsonl
date >> $O/summary3.txt
