#!/bin/bash
cd /opt/code/wt-mm
R=docs/results/2026-10-25-mm-metal-fp8-mmt/raw
echo "== noop $(date +%H:%M:%S)"
scripts/mm/noop.sh .mm/intarsia-main-0b84f499 .mm/intarsia-mmt $R/noop
echo "== speed $(date +%H:%M:%S)"
MAIN_LABEL=mm2-main-0b84f499 BRANCH_LABEL=mm2-fp8-mmt scripts/mm/speedslot.sh .mm/intarsia-main-0b84f499 .mm/intarsia-mmt .mm/bmr-main-0b84f499 .mm/bmr-mmt2 $R/speed
echo "== done $(date +%H:%M:%S)"
