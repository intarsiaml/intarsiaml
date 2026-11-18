#!/bin/zsh
# run 1: 27B serve --max-batch 16, batched MTP (+batched verify), sampled (T0.7 top_p .95 seeds 1234+i), knob off/on ABBA. Outside any slot; each round takes its own fleetq slot.
# --spec-off-at is capped at 9 by the CLI, so with 16 streams drafting only happens when <= 9 decode (tail of a wave); N=8 streams draft throughout.
W=${0:A:h:h:h}; L=$W/docs/results/2026-11-18-mgdnvsoak/raw; mkdir -p $L
M=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4
for cfg in "16 9 1" "8 8 1" "16 9 2" "8 8 2" "16 9 3" "8 8 3"; do
  set -- ${=cfg}
  MODEL=$M MT=27b SAMPLED=1 SOA=9 MB=16 N=$1 BMTP=$2 R=$3 MAXTOK=300 fleetq run --host endor --expect 12 --mem 45 --quiet gpu,cpu --label "mgdnvsoak run1 27b mb16 n$1 sampled r$3" -- zsh $W/scripts/mgdnvsoak/batch.sh >> $L/run1.log 2>&1
done
echo RUN1DONE >> $L/run1.log
