#!/bin/bash
# short check 4: 8B and LFM2 batched gates up to 16 rows
set -u
[ -n "${FLEETQ_JOB:-}" ] || exit 2
W=/opt/code/intarsiaml/.worktrees/mdec; O=$W/.lane/s4; mkdir -p $O; D=$W/.lane/target/release/deps
BM=$(ls -t $D/batch_metal-* | grep -v '\.d$' | head -1); BR=$(ls -t $D/batch_metal_real-* | grep -v '\.d$' | head -1)
INTARSIA_TEST_REAL_MODEL=1 INTARSIA_REQUIRE_GPU=1 $BM --nocapture > $O/dense16.log 2>&1; echo "batch_metal 8B exit $?"; grep -E 'bitwise|test result' $O/dense16.log
INTARSIA_LFM2_DIR=$HOME/.cache/huggingface/hub/models--LiquidAI--LFM2.5-1.2B-Instruct/snapshots/0f604ada3f766f9f257460c4c9f0b5d6f69d431b INTARSIA_REQUIRE_GPU=1 \
  $BR real_lfm2_batched_decode_is_bitwise_the_solo_step --exact --nocapture > $O/lfm2.log 2>&1; echo "lfm2 exit $?"; grep -E 'bitwise|test result|panicked' $O/lfm2.log
