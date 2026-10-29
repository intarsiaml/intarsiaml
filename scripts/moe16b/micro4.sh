#!/usr/bin/env bash
# Lane moe16b, micro job (< 3 min, not a window): the 35B batched decode step time at B = 8 / 16 (batch_cuda_real
# `timing_35b_batched_step`, graphs on), one process per side-stream variant (the knobs are read once per process).
#   fleetq run --host spark1 --quiet gpu,cpu --mem 35 --expect 4 --label "moe16b: step timing (micro)" -- scripts/moe16b/micro4.sh <out> <name>=<ENV=V,ENV=V> ...
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
OUT=/opt/code/intarsiaml/.worktrees/moe16b/docs/results/2026-10-28-moe16b-step-overlap/raw/$1; shift; mkdir -p "$(dirname "$OUT")"
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
for v in "$@"; do
  name=${v%%=*}; envs=${v#*=}; envs=${envs//,/ }
  r=$(ssh -n -o BatchMode=yes spark1 "export PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH; cd ~/intarsiaml-moe16b/src && T=\$(ls -t target/release/deps/batch_cuda_real-* | grep -v '\.d\$' | head -1); env $envs INTARSIA_QWEN36_DIR=$M35 INTARSIA_REQUIRE_GPU=1 timeout 80 \$T timing_35b_batched_step --exact --ignored --test-threads=1 --nocapture 2>&1 | grep -aE '^STEP|panicked|rror' | cut -c1-200")
  echo "$r" | sed "s/^/$name ($envs) /" | tee -a "$OUT"
done
