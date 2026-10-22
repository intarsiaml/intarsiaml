#!/bin/zsh
# mdl lane: bitwise gates + interleaved A/B speed of the Metal decode folds, ONE exclusive endor slot.
# usage: gates.sh <main-binary> <head-binary> <out-dir>      (main = origin/main db66c31a, head = lane-mdl/decode-levers)
MAIN=$1; HEAD=$2; O=$3; HERE=${0:A:h}; STAGES=${STAGES:-ids,speed,disp}   # which parts: ids (bitwise), speed (A/B), disp (dispatches/token)
M35=/Users/josef/models/Qwen3.6-35B-A3B-NVFP4; M8=/Users/josef/models/Qwen3-8B
PROMPT=$(cat $HERE/prompt-1k.txt); mkdir -p $O; S=$O/summary.txt; date >> $S
ids() { tag=$1; bin=$2; model=$3; shift 3   # greedy ids: stdout is completion + ids, stderr the stats
  $bin generate -m $model --backend metal --profile-every 0 --temperature 0 --ignore-eos --show-ids "$@" > $O/$tag.out 2> $O/$tag.err
  echo "$tag: $(grep -E 'finish=' $O/$tag.err | tail -1)" | tee -a $S; }
cmpids() { a=$1; b=$2; if [ -s $O/$a.out ] && cmp -s $O/$a.out $O/$b.out; then echo "IDENTICAL $a == $b ($(wc -c < $O/$a.out) bytes)" | tee -a $S; else echo "DIFFER $a vs $b" | tee -a $S; fi; }
[[ $STAGES == *ids* ]] && {
# 1. bitwise: 256 greedy ids, 35B short (chat) and ~4.4k prompt, 8B short (dense no-op)
for v in main head; do
  eval bin=\$$(echo $v | tr a-z A-Z)
  ids ids35s-$v $bin $M35 --max-seq 4096 --chat --no-think -p "Explain in three sentences why the sky is blue." --max-tokens 256
  ids ids35l-$v $bin $M35 --max-seq 8192 -p "$(cat $HERE/prompt4k.txt)" --max-tokens 256
  ids ids8b-$v $bin $M8 --max-seq 4096 --chat --no-think -p "Explain in three sentences why the sky is blue." --max-tokens 256
done
cmpids ids35s-main ids35s-head; cmpids ids35l-main ids35l-head; cmpids ids8b-main ids8b-head
}
[[ $STAGES == *speed* ]] && {
# 2. speed: decode tok/s at 1k context, 128 tokens, interleaved main/head x3 (generate: engine-reported decode tok/s)
for i in 1 2 3; do for v in main head; do
  eval bin=\$$(echo $v | tr a-z A-Z)
  ids spd-$v-$i $bin $M35 --max-seq 8192 -p "$PROMPT" --max-tokens 129
done; done
}
[[ $STAGES == *disp* ]] && {
# 3. dispatches per token: command-buffer profile, 65 vs 129 tokens, sum of dispatches (nd) -> (delta) / 64
for v in main head; do
  eval bin=\$$(echo $v | tr a-z A-Z)
  for n in 65 129; do
    INTARSIA_METAL_PROFILE=cb:$O/cb-$v-$n.jsonl $bin generate -m $M35 --backend metal --profile-every 0 --temperature 0 --ignore-eos --max-seq 8192 -p "$PROMPT" --max-tokens $n > $O/cb-$v-$n.out 2> $O/cb-$v-$n.err
  done
  python3 - $O $v >> $S <<'PY'
import json, sys, os
O, v = sys.argv[1:3]
def tot(n):
    return sum(json.loads(l)["nd"] for l in open(f"{O}/cb-{v}-{n}.jsonl") if l.strip())
a, b = tot(65), tot(129)
print(f"dispatches/token {v}: {(b - a) / 64:.2f}   (sum nd {a} -> {b})")
PY
done
}
date >> $S
