#!/bin/bash
cd /opt/code/wt-mm
R=docs/results/2026-10-25-mm-metal-fp8-mmt/raw/speed
for rep in 2 3 4; do
  scripts/mm/speed.sh .mm/intarsia-main-0b84f499 mm2-main-0b84f499-r$rep > $R/localeval-main-r$rep.log 2>&1
  scripts/mm/speed.sh .mm/intarsia-mmt mm2-fp8-mmt-r$rep > $R/localeval-branch-r$rep.log 2>&1
done
echo "== done $(date +%H:%M:%S)"
