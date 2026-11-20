#!/usr/bin/env bash
# kvonce: run CUDA backend test binaries built by build.sh (TESTS=...) on spark2, one at a time, GPU required.
#   FLEETQ_OWNER=claude-kvonce fleetq run --host spark2 --quiet gpu,cpu --mem 20 --expect 30 --label "kvonce tests" -- bash scripts/kvonce/tests.sh kv_once verify_attn_rows ...
# ENV="A=1 B=2" is passed to every binary (e.g. INTARSIA_ATTN_ROWS_PERSIST=0, KVONCE_TIMING=1); FILTER selects test names.
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
L='~/intarsiaml-kvonce'; rc=0
for t in "$@"; do
  echo "== $t $(date +%T) env='${ENV:-}'"
  ssh -n -o BatchMode=yes spark2 "cd $L/src/crates/intarsia-backend-cuda && env INTARSIA_REQUIRE_GPU=1 ${ENV:-} $L/bin/$t-test --test-threads 1 --nocapture ${FILTER:-} 2>&1 | grep -vE '^\s*$' | tail -${TAIL:-30}; exit \${PIPESTATUS[0]}"
  r=$?; echo "== $t exit $r"; [[ $r != 0 ]] && rc=$r
done
exit $rc
