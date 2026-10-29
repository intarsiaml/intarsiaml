#!/usr/bin/env bash
# Lane moe16b, micro job (< 3 min, not a window): the 35B batched decode step is bitwise the solo step with the side streams on
# (batch_cuda_real `real_35b_batched_decode_is_bitwise_the_solo_step` alone, graphs as in serve).
#   fleetq run --host spark1 --quiet gpu,cpu --mem 35 --expect 4 --label "moe16b: batch_cuda_real decode (micro)" -- scripts/moe16b/micro3.sh <out-file> [ENV=V ...]
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
OUT=/opt/code/intarsiaml/.worktrees/moe16b/docs/results/2026-10-28-moe16b-step-overlap/raw/$1; shift
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
ssh -n -o BatchMode=yes spark1 "export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH; cd ~/intarsiaml-moe16b/src && T=\$(ls -t target/release/deps/batch_cuda_real-* | grep -v '\.d\$' | head -1); date +%T; env $* INTARSIA_QWEN36_DIR=$M35 INTARSIA_REQUIRE_GPU=1 timeout 170 \$T real_35b_batched_decode_is_bitwise_the_solo_step --exact --test-threads=1 --nocapture 2>&1 | grep -aE '^test |test result|panicked|FAIL|identical|pairs|rror|Err' | cut -c1-400 | tail -20; date +%T" 2>&1 | tee "$OUT"
