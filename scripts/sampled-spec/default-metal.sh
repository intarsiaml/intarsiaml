#!/usr/bin/env bash
# Gates of the default-on change on Metal (one exclusive endor slot): default-metal.sh <main binary> <branch binary> <out dir>.
# INTARSIA_SPEC_SAMPLED unset on both binaries: plan show 27B / 35B / 8B (text, json), greedy ids 27B (MTP default) and 35B
# (--spec mtp --spec-k 2) short and 4K natural prompt, and the default check: a T 0.7 request runs verify passes on the branch
# (startup line + spec counters) and none on main.
set -uo pipefail
unset INTARSIA_SPEC_SAMPLED
base=$1; br=$2; O=$3; mkdir -p "$O"; : > $O/summary.txt
M27=~/models/Qwen3.8-27B-Quark-AWQ-MXFP4; M35=~/models/Qwen3.6-35B-A3B-NVFP4; M8=~/models/Qwen3-8B
P4K="$(cat /opt/code/deepeval/tests/data/prompts/natural-v1/n4096.txt)"
PS="Write a short paragraph about the history of the printing press."
for tag in base br; do
  bin=$base; [ $tag = br ] && bin=$br
  for m in 27:$M27 35:$M35 8:$M8; do
    $bin plan show --backend metal --model ${m#*:} --max-seq 8192 > $O/plan-$tag-${m%%:*}.txt 2> $O/plan-$tag-${m%%:*}.err
    $bin plan show --backend metal --model ${m#*:} --max-seq 8192 --json > $O/plan-$tag-${m%%:*}.json 2>> $O/plan-$tag-${m%%:*}.err
  done
  for m in 27:$M27: 35:$M35:--spec\ mtp\ --spec-k\ 2; do
    IFS=: read -r name dir extra <<< "$m"
    for p in short 4k; do
      prompt=$PS; [ $p = 4k ] && prompt=$P4K
      $bin generate --backend metal --model $dir --max-seq 8192 $extra --temperature 0 --max-tokens 96 --ignore-eos --show-ids \
        --prompt "$prompt" > $O/ids-$tag-$name-$p.out 2> $O/ids-$tag-$name-$p.err
    done
    $bin generate --backend metal --model $dir --max-seq 8192 $extra --temperature 0.7 --seed 3 --max-tokens 96 --prompt "$PS" \
      > $O/sampled-$tag-$name.out 2> $O/sampled-$tag-$name.err
  done
done
for f in $O/plan-base-*.txt $O/plan-base-*.json; do b=${f/plan-base-/plan-br-}; cmp -s $f $b && echo "same $(basename $f)" || echo "DIFF $(basename $f)"; done >> $O/summary.txt
for f in $O/ids-base-*.err; do b=${f/ids-base-/ids-br-}; [ -n "$(grep '^ids:' $f)" ] && cmp -s <(grep '^ids:' $f) <(grep '^ids:' $b) && echo "ids same $(basename $f)" || echo "ids DIFF $(basename $f)"; done >> $O/summary.txt
for f in $O/sampled-*.err; do echo "$(basename $f): $(grep -h 'sampled requests' $f) | $(grep -h '^spec: verify' $f)"; done >> $O/summary.txt
cat $O/summary.txt
