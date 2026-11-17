#!/bin/zsh
# mherd gate (herd), Metal: ABBA, `serve --max-batch 8 --prefix-store on --spec off`, A = herd off, B = INTARSIA_PREFIX_HERD=1.
# Per visit (fresh server and store): warm, herd8 seeds 1 2 (8 at once, one new SHARED-token prefix + 200 own tokens), distinct (8 x 8192), solo seeds 1 2 (the herd8 prompts one at a time).
# usage: [MODEL=35b] [SHARED=16448] [OWN=200] [NSEEDS="1 2"] herd.sh "A1 B1 B2 A2 ..."
source ${0:A:h}/lib.sh
VISITS=${1:-"A1 B1 B2 A2"}; SHARED=${SHARED:-16448}; OWN=${OWN:-200}; SEEDS=${SEEDS:-"1 2"}; MAXSEQ=${MAXSEQ:-17408}; OUT=${OUT:-$R/herd-$MODEL}; mkdir -p $OUT
for V in ${=VISITS}; do
  E=-; [ ${V[1]} = B ] && E=INTARSIA_PREFIX_HERD=1
  D=$OUT/$V; echo "== visit $V env $E $(cool)"
  start $D $E --max-seq $MAXSEQ --max-batch 8 --prefix-store on --spec off || continue
  C="python3 $S/../herd/herd_bench.py --url $URL --model intarsia --arm ${V[1]} --visit $V --shared $SHARED --own $OWN --out $OUT/results.jsonl"
  ${=C} --mode distinct --seed 900 --n 2 --distinct-len 512 --max-tokens 8 --out /dev/null > /dev/null 2>&1
  for s in ${=SEEDS}; do ${=C} --mode herd8 --seed $s; done
  ${=C} --mode distinct --seed 11 --distinct-len 8192
  for s in ${=SEEDS}; do ${=C} --mode solo --seed $s; done
  curl -sf $URL/v1/intarsia/prefix-store > $D/prefix-store.json
  echo "rss MB $(ps -o rss= -p $TP | awk '{print int($1/1024)}')"
  stop $D
done
echo "## done"
