#!/bin/zsh
# mmoe16 speed gate: 4 rounds, each its own fleetq slot (ab.sh reverses the arm order on even rounds).
cd ${0:A:h:h:h}
for r in 1 2 3 4; do
  fleetq run --host endor --quiet gpu,cpu --mem 40 --expect 16 --label "mmoe16: A/B round $r" -- scripts/mmoe16/round.sh $r > docs/results/2026-11-18-mmoe16/raw/ab-run-r$r.log 2>&1
done
echo AB_ALL_DONE >> docs/results/2026-11-18-mmoe16/raw/ab-run-r4.log
