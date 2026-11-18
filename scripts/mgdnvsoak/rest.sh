#!/bin/zsh
# runs 3, 4, 2 after run 1 (each its own fleetq slot, endor, --quiet gpu,cpu)
W=${0:A:h:h:h}; L=$W/docs/results/2026-11-18-mgdnvsoak/raw; S=$W/scripts/mgdnvsoak
M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4; M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4
until grep -q RUN1DONE $L/run1.log; do sleep 10; done
fq() { fleetq run --host endor --quiet gpu,cpu "$@"; }
fq --expect 12 --mem 40 --label "mgdnvsoak run3 27b thinking ids" -- zsh $S/run3.sh >> $L/run3.log 2>&1
DUR=1200 fq --expect 28 --mem 45 --label "mgdnvsoak run4 27b 20-min soak" -- zsh $S/run4.sh >> $L/run4.log 2>&1
for r in 1 2; do
  MODEL=$M27 MT=27b R=$r fq --expect 8 --mem 40 --label "mgdnvsoak run2 27b long ctx r$r" -- zsh $S/lc.sh >> $L/run2.log 2>&1
  MODEL=$M35 MT=35b R=$r fq --expect 8 --mem 50 --label "mgdnvsoak run2 35b long ctx r$r" -- zsh $S/lc.sh >> $L/run2.log 2>&1
done
echo RESTDONE >> $L/run2.log
