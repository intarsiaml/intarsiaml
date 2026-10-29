#!/bin/bash
# chunk4k gates: batch_recurrent, batch_metal_real 35B (+ the ignored long-prompt test), then cargo test --workspace --release.
W=/opt/code/intarsiaml/.worktrees/chunk4k; cd $W; O=$W/.lane/w3; mkdir -p $O
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer CARGO_TARGET_DIR=$W/.lane/target
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $O/job.log; }
export INTARSIA_TEST_REAL_MODEL=1 INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$HOME/Models/Qwen3.6-35B-A3B-NVFP4
cargo test --release --locked -p intarsia-models --test batch_recurrent -- --test-threads=1 > $O/batch_recurrent.log 2>&1; log "batch_recurrent exit $?: $(grep -E '^test result' $O/batch_recurrent.log)"
cargo test --release --locked -p intarsia-models --test batch_metal_real -- --test-threads=1 > $O/batch_metal_real.log 2>&1; log "batch_metal_real exit $?: $(grep -E '^test result' $O/batch_metal_real.log)"
cargo test --release --locked -p intarsia-models --test batch_metal_real -- --ignored real_35b_batched_engine_long_prompts_equal_the_local_engine --exact --nocapture > $O/long.log 2>&1; log "long prompts exit $?: $(grep -E '^test result' $O/long.log)"
unset INTARSIA_TEST_REAL_MODEL INTARSIA_QWEN36_DIR
cargo test --workspace --release --locked --no-fail-fast > $O/ws.log 2>&1; log "workspace exit $?: $(grep -E '^test result' $O/ws.log | awk '{p+=$4; f+=$6; i+=$8} END {print p" passed, "f" failed, "i" ignored"}')"
grep -E "^test .* FAILED|^failures:" $O/ws.log | head -10 | tee -a $O/job.log
log done
