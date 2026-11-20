#!/bin/zsh
# copyflipm 27B speed A/B: 4 rounds, each its own fleetq slot (`--quiet gpu,cpu`), FLEETQ_OWNER must be set. Run with nohup; logs under raw/.
cd ${0:A:h:h:h}
mkdir -p docs/results/2026-11-18-copyflipm/raw
for r in 1 2 3 4; do
  fleetq run --host endor --quiet gpu,cpu --mem 40 --expect 60 --label "copyflipm: 27B A/B round $r" -- env R0=$r ROUNDS=1 REPS=3 scripts/copyflipm/ab27.sh > docs/results/2026-11-18-copyflipm/raw/ab27-run-r$r.log 2>&1
done
echo AB27_ALL_DONE >> docs/results/2026-11-18-copyflipm/raw/ab27-run-r4.log
