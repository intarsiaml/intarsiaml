#!/usr/bin/env bash
# Lane step16 micro (< 5 min, fleetq --quiet gpu,cpu): rows_exact (every form, M = 1..16) on the dev binary under each env
# variant in EXACT, then the evicted row-kernel timing (fp8 only by default) for each variant in VARIANTS, a list of
# `bin:ENV=V,ENV=V` (bin = base | dev; `-` = no env), run in the given order (repeat entries to interleave). Prints `EVT ...`.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
L=/home/nvidia/intarsiaml-step16/bin
EXACT="${EXACT:--}"
VARIANTS="${VARIANTS:-base:- dev:- dev:- base:-}"
ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsiaml-step16/src/crates/intarsia-backend-cuda; date +%T;
  for e in $EXACT; do [ \"\$e\" = - ] && e=; echo \"== exact \$e\"; env \$(echo \$e | tr , ' ') INTARSIA_REQUIRE_GPU=1 timeout 80 $L/rows_exact-dev --test-threads=1 2>&1 | grep -aE '^test |test result|panicked|mismatch' | head -40; date +%T; done
  for v in $VARIANTS; do b=\${v%%:*}; e=\${v#*:}; [ \"\$e\" = - ] && e=; echo \"== \$v\"; env \$(echo \$e | tr , ' ') MSD2_ONLY=${ONLY:-fp8} timeout 40 $L/rows_exact-\$b --ignored --nocapture --test-threads=1 timing_rows_evict 2>&1 | grep -aE '^EVT|panicked|test result'; date +%T; done"
