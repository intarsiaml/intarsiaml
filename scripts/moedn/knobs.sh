#!/usr/bin/env bash
# Lane moedn: short check (< 3 min) on spark1: the MoE rows_exact gate (moe_decode_rows_are_bitwise_the_one_token_kernel)
# under every dedup down-kernel geometry (INTARSIA_MOE_DOWN_DD 8 / 16 / 32 x INTARSIA_MOE_DOWN_DD_R 2 / 4 / 8) and DD = 0 (v2).
#   fleetq run --host spark1 --quiet gpu,cpu --mem 8 --expect 3 --label "moedn: knob sweep" -- scripts/moedn/knobs.sh
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
L='~/intarsiaml-moedn'
RAW=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../docs/results/2026-10-29-moedn-down-proj/raw; mkdir -p "$RAW"
ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; cd $L/src/crates/intarsia-backend-cuda
for c in 0:4 8:2 8:4 8:8 16:2 16:4 16:8 32:2 32:4 32:8; do
  printf 'DD=%s DD_R=%s: ' \${c%%:*} \${c#*:}
  INTARSIA_MOE_DOWN_DD=\${c%%:*} INTARSIA_MOE_DOWN_DD_R=\${c#*:} INTARSIA_REQUIRE_GPU=1 timeout 60 $L/bin/rows_exact-dev --test-threads=1 moe_decode_rows 2>&1 | grep -aE 'test result|panicked|G-B FAIL' | tr '\n' ' '; echo
done" | tee "$RAW/knobs.txt"
