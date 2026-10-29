#!/usr/bin/env bash
# Lane attn16 short check (< 3 min, fleetq --quiet gpu,cpu on spark2): row-table attention == solo (row_tables G-B) and
# timing_attn_rows (ATTNT, us per launch at the 35B shape) for each INTARSIA_ATTN_ROWS_PERSIST value in $VARS, one
# process each.
#   fleetq run --host spark2 --quiet gpu,cpu --mem 8 --expect 3 --label "attn16: micro" -- scripts/attn16/micro.sh [bin]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
B=${1:-dev}; L=/home/nvidia/intarsiaml-attn16/bin
ssh -n -o BatchMode=yes spark2 "export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsiaml-attn16/src/crates/intarsia-backend-cuda; ${EXTRA:+export $EXTRA}
for v in ${VARS:-0 1}; do
  export INTARSIA_ATTN_ROWS_PERSIST=\$v
  [ ${GB:-1} = 1 ] && INTARSIA_REQUIRE_GPU=1 timeout 60 $L/row_tables-$B --exact attn_decode_rows_are_the_solo_decode --test-threads=1 --nocapture 2>&1 | grep -aoE 'G-B.*|panicked.*|test result.*' | sed \"s/^/\$v /\"
  for p in ${POSS:-1100 64}; do
    ATTNT_POS=\$p ATTNT_M=${ATTNT_M:-1,4,8,16} timeout 40 $L/row_tables-$B --ignored --exact timing_attn_rows --test-threads=1 --nocapture 2>&1 | grep -aoE 'ATTNT.*|panicked.*' | sed \"s/^/\$v /\"
  done
done"
