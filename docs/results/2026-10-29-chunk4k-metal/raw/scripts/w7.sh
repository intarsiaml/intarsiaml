#!/bin/bash
W=/opt/code/intarsiaml/.worktrees/chunk4k; cd $W; O=$W/.lane/w7; mkdir -p $O
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer CARGO_TARGET_DIR=$W/.lane/target
cargo test --release --locked -p intarsia-cli --test docs_env_vars 2>&1 | grep -E "^test result|error"
t0=$SECONDS
INTARSIA_TEST_REAL_MODEL=1 INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$HOME/Models/Qwen3.6-35B-A3B-NVFP4 \
  cargo test --release --locked -p intarsia-models --test batch_metal_real -- --ignored real_35b_batched_engine_long_prompts_equal_the_local_engine --exact --nocapture > $O/long-default.log 2>&1
echo "long test exit $? ($((SECONDS-t0)) s): $(grep -a -E '^test result' $O/long-default.log) | $(grep -a -c IDENTICAL $O/long-default.log) IDENTICAL, $(grep -a -c DIFFERS $O/long-default.log) DIFFERS"
