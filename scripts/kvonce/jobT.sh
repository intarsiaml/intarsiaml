#!/usr/bin/env bash
# kvonce tests: the bitwise test, the existing attention suites with the knob unset and set, the microbench.
#   FLEETQ_OWNER=claude-kvonce fleetq run --host spark2 --quiet gpu,cpu --mem 20 --expect 45 --label "kvonce tests+timing" -- bash scripts/kvonce/jobT.sh
D="$(dirname "${BASH_SOURCE[0]}")"; rc=0
TAIL=12 bash "$D/tests.sh" kv_once || rc=1
TAIL=8 bash "$D/tests.sh" verify_attn_rows row_tables attention attn_f8kv || rc=1
ENV="INTARSIA_CUDA_ATTN_KV_ONCE=1" TAIL=8 bash "$D/tests.sh" verify_attn_rows row_tables attention attn_f8kv || rc=1
ENV="KVONCE_TIMING=1" FILTER=timing TAIL=20 bash "$D/tests.sh" kv_once || rc=1
echo "jobT rc=$rc"; exit $rc
