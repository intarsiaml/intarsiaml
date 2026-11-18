#!/bin/zsh
# mmoedec speed gate: 4 rounds, each its own fleetq slot (ab.sh reverses the arm order on even rounds).
cd ${0:A:h:h:h}
for r in 1 2 3 4; do
  fleetq run --host endor --quiet gpu,cpu --mem 40 --expect 14 --label "mmoedec: A/B round $r" -- scripts/mmoedec/round.sh $r > docs/results/2026-11-18-mmoedec/raw/ab-run-r$r.log 2>&1
done
echo AB_ALL_DONE >> docs/results/2026-11-18-mmoedec/raw/ab-run-r4.log
