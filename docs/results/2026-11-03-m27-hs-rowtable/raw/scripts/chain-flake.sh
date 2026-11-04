#!/bin/zsh
# flake hunt: 6 rounds x 12 requests at max-batch 2, rt then def, twice each (ABBA order), ids compared with def-mb2 from the identity chain
export FLEETQ_OWNER=claude-m27-hsr ROUNDS=6
O=docs/results/2026-11-03-m27-hs-rowtable/raw/ident; S=docs/results/2026-11-03-m27-hs-rowtable/raw/scripts/ident-rt.sh
for t in rt def def rt; do
  n=$((n+1))
  fleetq run --host endor --mem 25 --expect 25 --label "m27 rowtable flake hunt $t mb2 x6 rounds #$n" -- zsh $S $O flake$n-$t-mb2 $PWD/.scratch/intarsia.$t 2
done
