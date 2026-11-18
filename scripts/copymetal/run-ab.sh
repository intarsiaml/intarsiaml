#!/bin/zsh
# copymetal speed gate: 4 rounds, each its own fleetq slot.
cd ${0:A:h:h:h}
for r in 1 2 3 4; do
  fleetq run --host endor --quiet gpu,cpu --mem 40 --expect 30 --label "copymetal: A/B round $r" -- scripts/copymetal/round.sh $r > docs/results/2026-11-18-copymetal/raw/ab-run-r$r.log 2>&1
done
echo AB_ALL_DONE >> docs/results/2026-11-18-copymetal/raw/ab-run-r4.log
