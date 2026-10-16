#!/bin/zsh
# Compare the gate outputs of gates.sh programmatically: cmp_gates.sh <gates outdir> <new tag>
O=$1; TAG=$2; R=${0:A:h}; rc=0
echo "== serving path / Model::forward bitwise (base vs $TAG, each chunk)"
for C in 256 2048; do python3 $R/cmp_serving.py $O/serving/base-chunk$C.jsonl $O/serving/$TAG-chunk$C.jsonl "chunk $C base vs $TAG" || rc=1; done
echo "== new chunk 256 vs 2048 (the standing chunk-invariance claim)"
python3 $R/cmp_serving.py $O/serving/$TAG-chunk256.jsonl $O/serving/$TAG-chunk2048.jsonl "$TAG chunk 256 vs 2048" || rc=1
echo "== generate (CLI) greedy ids, 48 tokens"
for f in p257s1 p2049s1 p2049s2 p4097s1 p4097s2; do for C in 256 2048; do
  python3 $R/first_divergence.py $O/generate/base-$f-c$C.err $O/generate/$TAG-$f-c$C.err "$f chunk $C base vs $TAG"
done; done
echo "== accuracy logs, base vs $TAG (every line that differs)"
for f in $O/accuracy/base-*.log; do
  b=${f:t}; n=${b#base-}
  if diff -q $f $O/accuracy/$TAG-$n > /dev/null; then echo "$n: identical logs"; else echo "$n: DIFFER"; diff $f $O/accuracy/$TAG-$n | head -20; rc=1; fi
done
exit $rc
