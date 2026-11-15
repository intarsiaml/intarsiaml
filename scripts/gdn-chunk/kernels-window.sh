#!/usr/bin/env bash
# Lane gdn-chunk, gate 1: `intarsia kernels test` (HIP) for Op::GdnDeltaRule. ON tatooine, inside a fleetq lease with radiance down.
#   chunk-all   the chunked kernel on every case it is compiled for (INTARSIA_GDN_CHUNK_MIN=1: T = 1 .. 4096)
#   chunk-def   the chunked kernel with its default min-T
#   seq         the sequential kernels (knob off) on the sweep with the new cases
set -u
cd "$(dirname "$0")/../.." && . scripts/gdn-chunk/lib.sh
KT="kernels test --backend hip --model $M8 --op gdn_delta_rule -v"
E="INTARSIA_GDN_CHUNK=1 INTARSIA_GDN_CHUNK_MIN=1" run 0 600 chunk-all $KT --kernel gdn_delta_rule_bf16_chunk
E="INTARSIA_GDN_CHUNK=1" run 0 600 chunk-def $KT --kernel gdn_delta_rule_bf16_chunk
E="" run 0 900 seq $KT
log "done"
