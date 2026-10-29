#!/usr/bin/env bash
# Lane stream16: timing_35b_batched_step (batched decode step at B = $SB, default 1,4,8,16; graphs, device argmax; prints
# `STEP B=.. ms min median max`) for each arm `bin@ENV=V+ENV=V` (`bin@-` = no env) in the argument list, in that order, one process each.
#   fleetq run --host spark1 --quiet gpu,cpu --mem 35 --expect 15 --label "stream16: step timing" -- scripts/stream16/steptime.sh base@- dev@- ...
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
for v in "$@"; do
  B=${v%%@*}; e=${v#*@}; [ "$e" = - ] && e=
  echo "== $v $(date +%T)"
  ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; cd ~/intarsiaml-stream16/src/crates/intarsia-models && env $(echo "$e" | tr + ' ') INTARSIA_STEP_BATCH=${SB:-1,4,8,16} INTARSIA_QWEN36_DIR=$M35 INTARSIA_REQUIRE_GPU=1 timeout 230 ~/intarsiaml-stream16/bin/batch_cuda_real-$B --ignored --nocapture --test-threads=1 --exact timing_35b_batched_step 2>&1 | grep -aE '^STEP|panicked|test result|row-table path'"
done
echo "== done $(date +%T)"
