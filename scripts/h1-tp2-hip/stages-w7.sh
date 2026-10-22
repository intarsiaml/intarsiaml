#!/usr/bin/env bash
# Lane H1 window 7 (ON tatooine, radiance down): re-gate the PR head after merging origin/main (main moved to 4c72749,
# which changes core model.rs and the HIP kernels table). Trees: base2 = origin/main 4c72749, head = lane head 11563b5.
#  A (parallel): tp=1 no-op, 35B on GPU 0 and 8B on GPU 1: plan show, 256 ids short / 4k.
#  B: tp=2 head: 35B ids short / 4k with graphs, --no-graphs, graphs + fusion; 8B short the same three;
#     35B long-set dumps with graphs and with graphs + fusion (compared bitwise offline).
set -uo pipefail
export O=${1:?out dir}
source "$HOME/h1/lib.sh"
t35() { for t in base2 head; do plan $t 0 $t-35b $M35; gen $t 0 $t-35b-short $M35 short; gen $t 0 $t-35b-4k $M35 4k; done; }
t8() { for t in base2 head; do plan $t 1 $t-8b $M8; gen $t 1 $t-8b-short $M8 short; gen $t 1 $t-8b-4k $M8 4k; done; }
t35 & a=$!; t8 & b=$!; wait $a $b
for m in 35b 8b; do
  same plan-base2-$m.out plan-head-$m.out "tp=1 plan $m"
  for k in short 4k; do same gen-base2-$m-$k.ids gen-head-$m-$k.ids "tp=1 ids $m $k"; done
done
for spec in "35b short $M35" "35b 4k $M35" "8b short $M8"; do
  set -- $spec; m=$1 k=$2 mp=$3
  gen head 0,1 head-$m-tp2g-$k $mp $k $TP2
  gen head 0,1 head-$m-tp2ng-$k $mp $k $TP2 --no-graphs
  E="INTARSIA_TP_FUSION=1" gen head 0,1 head-$m-tp2gf-$k $mp $k $TP2
  same gen-head-$m-tp2ng-$k.ids gen-head-$m-tp2g-$k.ids "tp=2 ids $m $k: --no-graphs vs graphs"
  same gen-head-$m-tp2gf-$k.ids gen-head-$m-tp2g-$k.ids "tp=2 ids $m $k: graphs+fusion vs graphs"
done
acc head 0,1 head-35b-tp2g-long $M35 $LG $TP2 --ref-noise $NL
E="INTARSIA_TP_FUSION=1" acc head 0,1 head-35b-tp2gf-long $M35 $LG $TP2 --ref-noise $NL
log "done"
