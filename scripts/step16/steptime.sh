#!/usr/bin/env bash
# Lane step16: timing_35b_batched_step (B = 8 / 16 batched decode step, graphs, device argmax; prints `STEP B=.. ms a b c`) for each
# test binary in the argument list, in that order (repeat names to interleave), one process each.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 35 --expect 15 --label "step16: step timing" -- scripts/step16/steptime.sh base dev dev base
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
for b in "$@"; do
  echo "== $b $(date +%T)"
  ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsiaml-step16/src/crates/intarsia-models && INTARSIA_QWEN36_DIR=$M35 INTARSIA_REQUIRE_GPU=1 timeout 230 ~/intarsiaml-step16/bin/batch_cuda_real-$b --ignored --nocapture --test-threads=1 --exact timing_35b_batched_step 2>&1 | grep -aE '^STEP|panicked|test result'"
done
echo "== done $(date +%T)"
