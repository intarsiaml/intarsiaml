#!/bin/zsh
# mherd gate (sjf), Metal: ABBA, `serve --max-batch 4 --spec off`, A = fifo, B = INTARSIA_PREFILL_ORDER=shortest. Per visit: warm, pair seeds (LONG-token prompt at t=0, SHORT-token at t=2 s), then solo of the same prompts.
# usage: [MODEL=35b] [LONG=40000] [SHORT=1000] [SEEDS="1 2"] sjf.sh "A1 B1 B2 A2"
source ${0:A:h}/lib.sh
VISITS=${1:-"A1 B1 B2 A2"}; LONG=${LONG:-40000}; SHORT=${SHORT:-1000}; SEEDS=${SEEDS:-"1 2"}; MAXSEQ=${MAXSEQ:-$((LONG+1024))}; OUT=${OUT:-$R/sjf-$MODEL}; mkdir -p $OUT
for V in ${=VISITS}; do
  E=-; [ ${V[1]} = B ] && E=INTARSIA_PREFILL_ORDER=shortest
  D=$OUT/$V; echo "== visit $V env $E $(cool)"
  start $D $E --max-seq $MAXSEQ --max-batch 4 --spec off || continue
  C="python3 $S/sjf_pair.py --url $URL --long $LONG --short $SHORT --arm ${V[1]} --visit $V --out $OUT/results.jsonl"
  ${=C} --mode pair --seed 900 --long 3000 --short 300 --out /dev/null > /dev/null 2>&1
  for s in ${=SEEDS}; do ${=C} --mode pair --seed $s; done
  for s in ${=SEEDS}; do ${=C} --mode solo --seed $s; done
  echo "rss MB $(ps -o rss= -p $TP | awk '{print int($1/1024)}')"
  stop $D
done
echo "## done"
