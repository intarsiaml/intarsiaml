#!/bin/bash
# MPROF window 3: is the served steady-state step slower than the short harness step because the GPU runs hot / slow under
# sustained load? Same harness (real_35b_batch_step_real_traffic), one B, MM_STEPS 64 (2.5 s) vs long (30 s), interleaved twice.
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mprof; O=$1; mkdir -p $O
BR=$(ls -t $W/.lane/target/release/deps/batch_metal_real-* | grep -v '\.d$' | head -1)
export INTARSIA_TEST_REAL_MODEL=1 INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$HOME/models/Qwen3.6-35B-A3B-NVFP4
run() { local tag=$1 b=$2 steps=$3; echo "## $tag B=$b steps=$steps $(date +%T)"; env MM_BATCHES=$b MM_STEPS=$steps $BR --ignored real_35b_batch_step_real_traffic --nocapture --test-threads=1 > $O/$tag.log 2>&1; echo "exit $?"; grep REALTRAFFIC $O/$tag.log; }
for r in 1 2; do
  run b16-s64-r$r 16 64
  run b16-s800-r$r 16 800
  run b8-s64-r$r 8 64
  run b8-s1200-r$r 8 1200
done
echo "## done $(date +%T)"
