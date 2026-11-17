#!/bin/zsh
# mflip gate, Metal 35B, ABBA visits, fresh server per visit. A = features off, B = ENVB on.
# usage: CELL=mix|herd|stag|sjf [SPEC=off|default] [MB=8] [N=8] [ENVB="INTARSIA_PREFIX_HERD=1 ..."] [OUT=dir] flip.sh "A1 B1 B2 A2"
#  mix : store on, N herd prompts (new 16448 shared + 200 own) at t=0 + a 1000-token prompt at t=2 s; seeds 1 2; solo of the same prompts
#  herd: store on, N herd prompts at t=0 (no short); seeds 1 2; solo
#  stag: store on, N herd prompts GAP apart: gaps 0.5 (seeds 1 2) and 2 (seeds 3 4); solo of seeds 1..4
#  sjf : no store, LONG-token prompt at t=0 + SHORT-token at t=2 s (sjf_pair.py), seeds 1 2, solo
source ${0:A:h}/lib.sh
CELL=${CELL:?}; SPEC=${SPEC:-off}; MB=${MB:-8}; N=${N:-8}; LONG=${LONG:-40000}; SHORT=${SHORT:-1000}; OWN=${OWN:-200}; SHARED=${SHARED:-16448}
OUT=${OUT:-$R/$CELL-$MODEL-$SPEC-mb$MB}; mkdir -p $OUT
SPECARG=(); [ $SPEC = off ] && SPECARG=(--spec off)
for V in ${=1:-"A1 B1 B2 A2"}; do
  E=-; [ ${V[1]} = B ] && E=$ENVB
  D=$OUT/$V; echo "== visit $V env $E spec $SPEC $(cool)"
  if [ $CELL = sjf ]; then
    start $D $E --max-seq $((LONG+1024)) --max-batch $MB $SPECARG || continue
    C="python3 $S/../mherd/sjf_pair.py --url $URL --long $LONG --short $SHORT --arm ${V[1]} --visit $V --out $OUT/results.jsonl"
    ${=C} --mode pair --seed 900 --long 3000 --short 300 --out /dev/null > /dev/null 2>&1
    for s in 1 2; do ${=C} --mode pair --seed $s; done
    for s in 1 2; do ${=C} --mode solo --seed $s; done
  else
    start $D $E --max-seq 17408 --max-batch $MB --prefix-store on $SPECARG || continue
    C="python3 $S/burst.py --url $URL --arm ${V[1]} --visit $V --cell $CELL --shared $SHARED --own $OWN --out $OUT/results.jsonl"
    python3 $S/burst.py --url $URL --arm W --visit $V --shared 512 --own 20 --n 2 --seed 900 --max-tokens 8 --mode burst --out /dev/null > /dev/null 2>&1
    case $CELL in
      mix)  for s in 1 2; do ${=C} --mode burst --seed $s --n $N --short-len $SHORT --short-at 2; done
            for s in 1 2; do ${=C} --mode solo --seed $s --n $N --short-len $SHORT; done ;;
      herd) for s in 1 2; do ${=C} --mode burst --seed $s --n $N; done
            for s in 1 2; do ${=C} --mode solo --seed $s --n $N; done ;;
      stag) for s in 1 2; do ${=C} --mode burst --seed $s --n $N --gap 0.5; done
            for s in 3 4; do ${=C} --mode burst --seed $s --n $N --gap 2; done
            for s in 1 2 3 4; do ${=C} --mode solo --seed $s --n $N; done ;;
    esac
    curl -sf $URL/v1/intarsia/prefix-store > $D/prefix-store.json
  fi
  echo "rss MB $(ps -o rss= -p $TP | awk '{print int($1/1024)}')"
  grep -i "spec\|mtp\|herd\|prefill order\|prefix store" $D/server.err 2>/dev/null | head -12 > $D/startup-lines.txt
  stop $D
done
echo "## done"
