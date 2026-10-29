#!/usr/bin/env bash
# Lane moedn: timing_35b_batched_step (B = 8 / 16 batched decode step, graphs, device argmax; prints `STEP B=.. ms a b c`) of the
# dev test binary under each env variant in the argument list (`ENV=V,ENV=V`, `-` = none), in that order, one process each.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 35 --expect 15 --label "moedn: step timing" -- scripts/moedn/steptime.sh - INTARSIA_GDN_DEC=0 ...
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
B=${BIN:-dev}
for v in "$@"; do
  e=$v; [ "$e" = - ] && e=
  echo "== $v $(date +%T)"
  ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsiaml-moedn/src/crates/intarsia-models && env $(echo "$e" | tr , ' ') INTARSIA_QWEN36_DIR=$M35 INTARSIA_REQUIRE_GPU=1 timeout 230 ~/intarsiaml-moedn/bin/batch_cuda_real-$B --ignored --nocapture --test-threads=1 --exact timing_35b_batched_step 2>&1 | grep -aE '^STEP|panicked|test result|row-table path'"
done
echo "== done $(date +%T)"
