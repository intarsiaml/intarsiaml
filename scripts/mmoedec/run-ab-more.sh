#!/bin/zsh
# mmoedec: extra rounds 5..10 of the speed gate (the first four were noisy, run-ab.sh), same round.sh, same order rule (even rounds reversed).
cd ${0:A:h:h:h}
for r in 5 6 7 8 9 10; do
  fleetq run --host endor --quiet gpu,cpu --mem 40 --expect 14 --label "mmoedec: A/B round $r" -- scripts/mmoedec/round.sh $r > docs/results/2026-11-18-mmoedec/raw/ab-run-r$r.log 2>&1
done
echo AB_ALL_DONE >> docs/results/2026-11-18-mmoedec/raw/ab-run-r10.log
