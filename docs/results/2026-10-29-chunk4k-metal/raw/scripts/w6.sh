#!/bin/bash
W=/opt/code/intarsiaml/.worktrees/chunk4k; cd $W; O=$W/.lane/w6; mkdir -p $O
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer CARGO_TARGET_DIR=$W/.lane/target
t0=$SECONDS
INTARSIA_TEST_REAL_MODEL=1 INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$HOME/Models/Qwen3.6-35B-A3B-NVFP4 \
  cargo test --release --locked -p intarsia-models --test batch_metal_real -- --ignored real_35b_batched_engine_long_prompts_equal_the_local_engine --exact --nocapture > $O/long-default.log 2>&1
echo "long test exit $? ($((SECONDS-t0)) s): $(grep -a -E '^test result' $O/long-default.log)"
grep -a -E "LONGENG (reference|[0-9]-slot)|IDENTICAL|DIFFERS" $O/long-default.log | cut -c1-170
t0=$SECONDS
INTARSIA_REQUIRE_GPU=1 cargo test --release --locked -p intarsia-backend-metal > $O/metal-crate.log 2>&1
echo "backend-metal tests exit $? ($((SECONDS-t0)) s): $(grep -a -E '^test result' $O/metal-crate.log | awk '{p+=$4; f+=$6; i+=$8} END {print p" passed, "f" failed, "i" ignored"}')"
