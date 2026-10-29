#!/usr/bin/env bash
# Lane moe16b, micro job (< 3 min each part, not a window): the fp8 rows exactness tests (incl. the balanced-split shapes) and the
# evicted micro-timing of the balanced-split shapes at M 8..16, on the rows_exact binary built from the synced tree.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 10 --expect 5 --label "moe16b: fp8 balanced-split micro" -- scripts/moe16b/micro2.sh <out>
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
OUT=/opt/code/intarsiaml/.worktrees/moe16b/docs/results/2026-10-28-moe16b-step-overlap/raw/$1; mkdir -p "$OUT"
rsh() { ssh -n -o BatchMode=yes spark1 "export PATH=/usr/local/cuda/bin:\$PATH FLEETQ_JOB=$FLEETQ_JOB INTARSIA_REQUIRE_GPU=1; cd ~/intarsiaml-moe16b/src && T=\$(ls -t target/release/deps/rows_exact-* | grep -v '\.d\$' | head -1); $*"; }
date +%T
rsh 'timeout 120 $T --test-threads=3 fp8_rows 2>&1 | grep -aE "^test |G-B|panicked|FAIL|result" | cut -c1-300' | tee "$OUT/exact.txt"
date +%T
for r in 1 2; do  # two passes: the spread between them is the noise
  rsh 'MSD2_ONLY=fp8bal timeout 80 $T --ignored --nocapture timing_rows_evict 2>&1 | grep -aE "^EVT|panicked" | grep -vE "M=(2|4) "' > "$OUT/evict-$r.txt"
  echo "pass $r: $(grep -c EVT "$OUT/evict-$r.txt") rows"
done
date +%T
