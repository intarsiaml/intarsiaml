#!/usr/bin/env bash
# Lane gdnx micro (< 5 min, fleetq --quiet gpu,cpu): the row_tables G-B tests for the GDN rows kernels under each EXACT env
# variant, then timing_gdn_rows (GDNT lines) for each VARIANTS entry `bin:ENV=V,ENV=V` (`-` = no env), in the given order.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
L=/home/nvidia/intarsiaml-gdnx/bin
EXACT="${EXACT:--}"
VARIANTS="${VARIANTS:-dev:- dev:INTARSIA_GDN_DEC=0}"
ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsiaml-gdnx/src/crates/intarsia-backend-cuda; date +%T;
  for e in $EXACT; do [ \"\$e\" = - ] && e=; echo \"== exact \$e\"; env \$(echo \$e | tr , ' ') INTARSIA_REQUIRE_GPU=1 timeout 100 $L/row_tables-dev --test-threads=1 gdn 2>&1 | grep -aE '^test |test result|panicked|G-B'; env \$(echo \$e | tr , ' ') INTARSIA_REQUIRE_GPU=1 timeout 100 $L/gdn_split-dev --test-threads=1 2>&1 | grep -aE '^test |test result|panicked'; date +%T; done
  for v in $VARIANTS; do b=\${v%%:*}; e=\${v#*:}; [ \"\$e\" = - ] && e=; echo \"== \$v\"; env \$(echo \$e | tr , ' ') GDNT_M=${GDNT_M:-1,4,8,16} timeout 60 $L/row_tables-\$b --ignored --nocapture --test-threads=1 timing_gdn_rows 2>&1 | grep -aE '^GDNT|panicked|test result'; date +%T; done"
