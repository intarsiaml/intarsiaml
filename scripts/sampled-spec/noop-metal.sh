#!/usr/bin/env bash
# Metal no-op gate of sampled speculation (docs/results/2026-11-05-sampled-spec.md, rule 4), inside one exclusive endor slot:
# noop-metal.sh <base binary> <branch binary> <out dir>. plan show (text, json) for the 27B / 35B / 8B, kernels test, greedy ids
# (branch with INTARSIA_SPEC_SAMPLED=1) for the 27B (MTP default) and the 35B (--spec mtp --spec-k 2), short and 4K natural
# prompt, the 8B HF fp32 goldens, and a sampled smoke run of both variants on the 27B.
set -uo pipefail
base=$1; br=$2; O=$3; mkdir -p "$O"
M27=~/models/Qwen3.8-27B-Quark-AWQ-MXFP4; M35=~/models/Qwen3.6-35B-A3B-NVFP4; M8=~/models/Qwen3-8B
G=/opt/code/intarsiaml/golden
P4K="$(cat /opt/code/deepeval/tests/data/prompts/natural-v1/n4096.txt)"
PS="Write a short paragraph about the history of the printing press."
for tag in base br; do
  bin=$base; [ $tag = br ] && bin=$br
  for m in 27:$M27 35:$M35 8:$M8; do
    $bin plan show --backend metal --model ${m#*:} --max-seq 8192 > $O/plan-$tag-${m%%:*}.txt 2> $O/plan-$tag-${m%%:*}.err
    $bin plan show --backend metal --model ${m#*:} --max-seq 8192 --json > $O/plan-$tag-${m%%:*}.json 2>> $O/plan-$tag-${m%%:*}.err
  done
  $bin kernels test --backend metal > $O/kernels-$tag.log 2>&1; echo "kernels $tag exit $?" >> $O/summary.txt
  for m in 27:$M27: 35:$M35:--spec\ mtp\ --spec-k\ 2; do
    IFS=: read -r name dir extra <<< "$m"
    for p in short 4k; do
      prompt=$PS; [ $p = 4k ] && prompt=$P4K
      env INTARSIA_SPEC_SAMPLED=$([ $tag = br ] && echo 1 || echo 0) $bin generate --backend metal --model $dir --max-seq 8192 $extra \
        --temperature 0 --max-tokens 96 --ignore-eos --show-ids --prompt "$prompt" > $O/ids-$tag-$name-$p.out 2> $O/ids-$tag-$name-$p.err
    done
  done
  for g in qwen3-8b-fp32 qwen3-8b-long-fp32; do
    $bin accuracy --backend metal --model $M8 --golden $G/$g.safetensors > $O/golden-$tag-$g.log 2>&1; echo "golden $tag $g exit $?" >> $O/summary.txt
  done
done
for v in reject match; do
  INTARSIA_SPEC_SAMPLED=$v INTARSIA_SPEC_TRACE=1 $br generate --backend metal --model $M27 --max-seq 8192 --temperature 0.7 --seed 3 \
    --max-tokens 128 --prompt "$PS" > $O/sampled-27-$v.out 2> $O/sampled-27-$v.err
done
for f in $O/plan-base-*; do b=${f/plan-base-/plan-br-}; cmp -s <(grep -v "loaded in\|elapsed\| ms\b" $f) <(grep -v "loaded in\|elapsed\| ms\b" $b) && echo "same $(basename $f)" || echo "DIFF $(basename $f)"; done >> $O/summary.txt
for f in $O/ids-base-*.out; do b=${f/ids-base-/ids-br-}; cmp -s <(grep -iv "decode\|prefill\|ms\b" $f) <(grep -iv "decode\|prefill\|ms\b" $b) && echo "same $(basename $f)" || echo "DIFF $(basename $f)"; done >> $O/summary.txt
cat $O/summary.txt
