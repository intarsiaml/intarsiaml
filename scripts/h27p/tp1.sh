#!/usr/bin/env bash
# Lane H27P: tp1 (card 0) check of the recipe change, ON tatooine INSIDE a GPU window
S="$(dirname "${BASH_SOURCE[0]}")"; source "$S/lib.sh"
echo "== tp1 (card 0): base vs dev, 4k-0 prefill x3 and ids (4k-0, 4k-3, 16k-3), --spec off"
t1=$HOME/intarsiaml-h27p-logs/tp1; mkdir -p $t1
for n in 4k-0 4k-3 16k-3; do
  case $n in 4k*) ms=8192 ;; *) ms=20480 ;; esac
  for arm in base dev; do
    bin=h27p-intarsia; [ $arm = base ] && bin=base-intarsia
    for rep in 1 2 3; do
      [ $rep -gt 1 ] && [ $n != 4k-0 ] && continue
      G=0 gpu 230 /work/$bin generate -m $M27 --backend hip --mem-budget 29 --max-seq $ms --spec off --chat --no-think --temperature 0 \
        --profile-every 0 --ignore-eos --show-ids -p "$(cat $R/scripts/cmtp/ids-prompts/$n.txt)" --max-tokens 128 > $t1/$n-$arm-$rep.out 2> $t1/$n-$arm-$rep.err
    done
    grep -a '^ids:' $t1/$n-$arm-1.err > $t1/$n-$arm.ids
    echo "tp1 $n $arm: $(grep -a 'prefill ' $t1/$n-$arm-*.err | grep -aoE 'prefill [0-9]+ ms \([0-9.]+ tok/s\)' | tr '\n' ' ') chunk: $(grep -ao 'prefill chunk [0-9]*' $t1/$n-$arm-1.err | head -1)"
  done
  cmp -s $t1/$n-base.ids $t1/$n-dev.ids && echo "tp1 $n ids identical" || echo "tp1 $n ids DIFFER"
done
