#!/bin/zsh
# mcopylen gate (c): 4 rounds, each its own fleetq slot (order reversed on even rounds by ab.sh): base / n20,7 / n20,15 / n20,15,a (adaptive).
cd ${0:A:h:h:h}
for r in 1 2 3 4; do
  fleetq run --host endor --quiet gpu,cpu --mem 40 --expect 30 --label "mcopylen: A/B round $r" -- env W=ab R0=$r ROUNDS=1 REPS=3 CELLS="copy3k json-xform agent-edit code-gen" \
    ARMS="base:X=1 n20L7:INTARSIA_SPEC_COPY=20,7 n20L15:INTARSIA_SPEC_COPY=20,15 n20L15a:INTARSIA_SPEC_COPY=20,15,a" scripts/mcopylen/ab.sh > docs/results/2026-11-18-mcopylen/raw/ab-run-r$r.log 2>&1
done
