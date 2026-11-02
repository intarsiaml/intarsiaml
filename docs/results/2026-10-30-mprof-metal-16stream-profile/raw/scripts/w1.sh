#!/bin/bash
# MPROF window 1: batched decode step, real-traffic harness (batch_metal_real::real_35b_batch_step_real_traffic),
# B = 1,4,8,9,12,16. usage: w1.sh OUTDIR  (inside a fleetq slot)
set -u
[ -n "${FLEETQ_JOB:-}" ] || { echo "not inside a fleetq slot" >&2; exit 2; }
W=/opt/code/intarsiaml/.worktrees/mprof; O=$1; mkdir -p $O
BR=$(ls -t $W/.lane/target/release/deps/batch_metal_real-* | grep -v '\.d$' | head -1)
export MM_BATCHES=1,4,8,9,12,16 MM_STEPS=32 INTARSIA_TEST_REAL_MODEL=1 INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$HOME/models/Qwen3.6-35B-A3B-NVFP4
run() { local tag=$1; shift; echo "## $tag $(date +%T)"; env X=1 "$@" $BR --ignored real_35b_batch_step_real_traffic --nocapture --test-threads=1 > $O/$tag.log 2>&1; echo "exit $?"; grep REALTRAFFIC $O/$tag.log; }
run plain1
run plain2
run cb INTARSIA_METAL_PROFILE=cb:$O/cb.jsonl
run ce1 INTARSIA_METAL_COMMIT_EVERY=1 INTARSIA_METAL_PROFILE=cb:$O/ce1.jsonl
run kernel INTARSIA_METAL_PROFILE=kernel:$O/kernel.jsonl
run plain3
ls -la $O
gzip -f $O/*.jsonl
echo "## done $(date +%T)"
