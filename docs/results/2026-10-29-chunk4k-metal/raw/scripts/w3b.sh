#!/bin/bash
# chunk4k gates after merging origin/main b26979329: long-prompt identity (3 cases, one process each), then workspace tests with no GPU-required env.
W=/opt/code/intarsiaml/.worktrees/chunk4k; cd $W; O=$W/.lane/w3b; mkdir -p $O
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer CARGO_TARGET_DIR=$W/.lane/target
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $O/job.log; }
for c in 0 1 2; do
  INTARSIA_TEST_REAL_MODEL=1 INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$HOME/Models/Qwen3.6-35B-A3B-NVFP4 INTARSIA_LONGENG_CASE=$c \
   cargo test --release --locked -p intarsia-models --test batch_metal_real -- --ignored real_35b_batched_engine_long_prompts_equal_the_local_engine --exact --nocapture > $O/long-$c.log 2>&1
  log "long-prompt case $c exit $?: $(grep -E '^test result' $O/long-$c.log) | $(grep -a -c 'IDENTICAL' $O/long-$c.log) IDENTICAL, $(grep -a -c 'DIFFERS' $O/long-$c.log) DIFFERS"
done
cargo test --workspace --release --locked --no-fail-fast > $O/ws.log 2>&1; log "workspace exit $?: $(grep -E '^test result' $O/ws.log | awk '{p+=$4; f+=$6; i+=$8} END {print p" passed, "f" failed, "i" ignored"}')"
grep -a -E "^test .* FAILED" $O/ws.log | head -10 | tee -a $O/job.log
log done
