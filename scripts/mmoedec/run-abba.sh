#!/bin/zsh
# mmoedec: A B B A rounds 11, 13, 15, 17 (round-abba.sh), each its own fleetq slot.
cd ${0:A:h:h:h}
for r in 11 13 15 17; do
  fleetq run --host endor --quiet gpu,cpu --mem 40 --expect 12 --label "mmoedec: ABBA round $r" -- scripts/mmoedec/round-abba.sh $r > docs/results/2026-11-18-mmoedec/raw/abba-run-r$r.log 2>&1
done
echo ABBA_ALL_DONE >> docs/results/2026-11-18-mmoedec/raw/abba-run-r17.log
