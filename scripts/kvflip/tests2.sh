#!/usr/bin/env bash
# kvflip: the flip build's CUDA test binaries on spark1: knob unset (= the new default on), =0, =1; plus row_tables with the knob unset and =0.
#   FLEETQ_OWNER=claude-kvflip fleetq run --host spark1 --load gpu --mem 20 --expect 15 --label "kvflip flip tests" -- bash scripts/kvflip/tests2.sh
D="$(dirname "${BASH_SOURCE[0]}")"
for e in "" "INTARSIA_CUDA_ATTN_KV_ONCE=0" "INTARSIA_CUDA_ATTN_KV_ONCE=1"; do
  echo "#### knob env: '$e'"
  ENV="$e" bash "$D/tests.sh" kv_once attention verify_attn_rows row_tables
done
echo "#### KVONCE_TIMING=1 kv_once (both tests in one process, no filter: the lock must serialise them)"
ENV="KVONCE_TIMING=1" TAIL=40 bash "$D/tests.sh" kv_once
