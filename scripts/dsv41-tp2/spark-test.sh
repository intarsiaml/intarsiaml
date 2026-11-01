#!/bin/bash
# Run one DSv4.1 test binary filter on a Spark against the rank caches there; run under `fleetq run`.
#
#   fleetq run --host spark2 --load cpu --mem 40 --expect 20 --label "dsv41 verify rows" -- \
#     scripts/dsv41-tp2/spark-test.sh spark2 dsv41_runtime verify_rows \
#       INTARSIA_DSV41_BACKEND=cpu INTARSIA_DSV41_CACHE='$HOME/intarsia-dsv41-tp2/cache/l3-r0of1'
#
# Arguments: host, test binary (in ~/intarsia-dsv41-graphs/bin, from scripts/dsv41-tp2/build.sh), test filter, then
# VAR=value pairs for the test (quote $HOME so it expands on the Spark). The full output goes to
# ~/intarsia-dsv41-graphs/logs/<binary>-<filter>-<time>.log there; the summary lines (DSPARK / SPEC / VERIFY / RUNTIME,
# test result, panics) come back here. Exit status: the test's.
#
# Caches the checks of 2026-11-01 used (docs/results/2026-11-01-dsv41-tp2.md):
#   dspark_rounds_match_the_official_drafter   INTARSIA_DSV41_CACHE=.../l3-r0of1 INTARSIA_DSV41_FIXTURES=$HOME/intarsia-dsv41-tp2/fixtures
#   speculative_greedy_equals_plain_greedy     INTARSIA_DSV41_CACHE=.../l3-r0of1
#   verify_rows_track_decode_through_rollback  INTARSIA_DSV41_CACHE=.../l3-r0of1 [INTARSIA_DSV41_VT_PROMPT=1100 INTARSIA_DSV41_VT_CAP=2048]
#   ..._two_ranks                              INTARSIA_DSV41_CACHE_R0=.../l3-r0of2 INTARSIA_DSV41_CACHE_R1=.../l3-r1of2 (spark1 has both)
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
[[ $# -ge 3 ]] || { echo "usage: $0 HOST BINARY FILTER [VAR=value ...]" >&2; exit 2; }
H=$1 BIN=$2 FILTER=$3
shift 3
ENVS="$*"
LOG="\$HOME/intarsia-dsv41-graphs/logs/$BIN-$FILTER-$(date +%m%d-%H%M%S).log"
ssh -n -o BatchMode=yes "$H" "export FLEETQ_JOB=$FLEETQ_JOB; mkdir -p \$HOME/intarsia-dsv41-graphs/logs; cat \$HOME/intarsia-dsv41-graphs/bin/STAMP; \
  env $ENVS timeout 3000 \$HOME/intarsia-dsv41-graphs/bin/$BIN $FILTER --nocapture --test-threads 1 > $LOG 2>&1; rc=\$?; \
  grep -aE '^(DSPARK|SPEC|VERIFY|RUNTIME)|skipped|test result|panicked' $LOG | grep -v 'VERIFY pos'; echo \"log $LOG\"; exit \$rc"
