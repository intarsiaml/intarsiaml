#!/bin/zsh
# Final ABAB speed: A = baseline binary (.scratch/intarsia.def), B = this lane (.scratch/intarsia.rt); both serve --max-batch MB --spec off --max-seq 18000.
# mb8: localeval speed --concurrency 8,16 (16 streams over 8 slots; --max-batch 16 is refused for the 27B on Metal), mb4: --concurrency 4,4; sizes 4096 then 16384, gen 512.
export FLEETQ_OWNER=claude-m27-hsr
O=docs/results/2026-11-03-m27-hs-rowtable/raw/speed; S=docs/results/2026-11-03-m27-hs-rowtable/raw/scripts/conc-arm.sh
for cfg in "8 8,16 40" "4 4,4 20"; do
  set -- ${=cfg}; mb=$1; conc=$2; ex=$3
  for t in def rt def rt; do
    n=$((n+1))
    fleetq run --host endor --quiet gpu,cpu --mem 45 --expect $ex --label "m27 rowtable ABAB mb$mb $t #$n" -- zsh $S $t-mb$mb-$n $PWD/.scratch/intarsia.$t $O $mb $conc
  done
done
