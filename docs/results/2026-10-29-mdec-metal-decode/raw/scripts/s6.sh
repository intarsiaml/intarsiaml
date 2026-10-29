#!/bin/bash
# build metal CLI + gate tests, then (inside the same slot) run 8B/LFM2/multirow gates
set -u
[ -n "${FLEETQ_JOB:-}" ] || exit 2
W=/opt/code/intarsiaml/.worktrees/mdec; cd $W; O=$W/.lane/s6; mkdir -p $O
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer CARGO_TARGET_DIR=$W/.lane/target
cargo build --release --locked -p intarsia-cli --features metal 2>&1 | grep -E '^error|Finished'
INTARSIA_TEST_REAL_MODEL=1 INTARSIA_REQUIRE_GPU=1 cargo test --release --locked -p intarsia-models --test batch_metal -- --nocapture > $O/dense16.log 2>&1; echo "batch_metal 8B exit $?"; grep -E 'pairs bitwise|test result' $O/dense16.log
INTARSIA_LFM2_DIR=$HOME/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/0f604ada3f766f9f257460c4c9f0b5d6f69d431b INTARSIA_REQUIRE_GPU=1 \
  cargo test --release --locked -p intarsia-models --test batch_metal_real -- real_lfm2_batched_decode_is_bitwise_the_solo_step --exact --nocapture > $O/lfm2.log 2>&1; echo "lfm2 exit $?"; grep -E 'batch 1[0-9]|pairs bitwise|test result|panicked' $O/lfm2.log
INTARSIA_REQUIRE_GPU=1 cargo test --release --locked -p intarsia-backend-metal --test multirow > $O/multirow.log 2>&1; echo "multirow exit $?"; grep -E 'test result|FAILED' $O/multirow.log
