#!/bin/zsh
# identity chain: def (baseline) and rt (this lane) at max-batch 2, 4, 8
export FLEETQ_OWNER=claude-m27-hsr
O=docs/results/2026-11-03-m27-hs-rowtable/raw/ident; S=docs/results/2026-11-03-m27-hs-rowtable/raw/scripts/ident-rt.sh
for mb in 2 4 8; do for t in def rt; do
  fleetq run --host endor --mem 35 --expect 8 --label "m27 rowtable ident $t mb$mb" -- zsh $S $O $t-mb$mb $PWD/.scratch/intarsia.$t $mb
done; done
