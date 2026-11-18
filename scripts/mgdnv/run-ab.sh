#!/bin/zsh
# mgdnv gate (c): 4 rounds, each its own fleetq slot (the arm order is reversed on even rounds by ab.sh).
cd ${0:A:h:h:h}
for r in 1 2 3 4; do
  fleetq run --host endor --quiet gpu,cpu --mem 40 --expect 12 --label "mgdnv: A/B round $r" -- scripts/mgdnv/round.sh $r > docs/results/2026-11-18-mgdnv/raw/ab-run-r$r.log 2>&1
done
