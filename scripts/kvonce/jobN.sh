#!/usr/bin/env bash
# kvonce: the bitwise test on the rebuilt binary, then kernel time per verify pass under nsys, knob off and on, sampled T=0.7 at 4k / 32k / 64k.
#   FLEETQ_OWNER=claude-kvonce fleetq run --host spark2 --quiet gpu,cpu --mem 55 --expect 50 --label "kvonce nsys" -- bash scripts/kvonce/jobN.sh
D="$(dirname "${BASH_SOURCE[0]}")"
TAIL=6 bash "$D/tests.sh" kv_once || exit 1
bash "$D/nsys.sh" off "" 0.7:4096 0.7:32768 0.7:65536
SERVE_ENV="INTARSIA_CUDA_ATTN_KV_ONCE=1" bash "$D/nsys.sh" on "" 0.7:4096 0.7:32768 0.7:65536
