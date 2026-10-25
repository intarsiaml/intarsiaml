#!/usr/bin/env bash
# Lane wg, ON tatooine inside a window (wg-win.sh), tree ~/intarsiaml-wg already built by prep.sh tests:
# the 35B batched step on HIP, three arms in one fresh process each: op listing (which kernels ran), the per-op step profile at B=1/4/16
# (each op synchronised: read the shares) and the unprofiled model-loop step time at B=1/2/4/8/16.
#   main    knob unset (registry = main)
#   a16     INTARSIA_HIP_WMMA_ROWS=1 (default table: W8A16 WMMA only; the 35B fp8 layers are all W8A8, so this is main's kernels)
#   opt1    INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all (option 1: every kind WMMA at all M)
set -uo pipefail
export HIP_VISIBLE_DEVICES=0
LOG=$HOME/wg-logs; mkdir -p "$LOG"
DEV="env INTARSIA_REPO=$HOME/intarsiaml-wg INTARSIA_TARGET_VOL=intarsia-cargo-target-wg $HOME/intarsiaml-wg/deploy/tatooine/dev.sh"
T="cargo test --release --locked -p intarsia-models --test batch_cuda_real -- --include-ignored --test-threads=1 --nocapture real_35b_batch_op_listing real_35b_batch_step_profile real_35b_batch_step_timing"
arm() {  # name, env assignments...
  local name=$1; shift
  echo "=== $name  ($*)  $(date +%T)"
  env "$@" $DEV env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=/models/Qwen3.6-35B-A3B-NVFP4 INTARSIA_TIMING_ROWS=1,2,4,8,16 $T > "$LOG/profile-$name.log" 2>&1
  echo "exit=$?  $(grep -E '^test result' "$LOG/profile-$name.log" | tr '\n' ' ')"
}
arm main INTARSIA_WG_ARM=main
arm a16 INTARSIA_HIP_WMMA_ROWS=1
arm opt1 INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all
echo PROFILE DONE
