#!/usr/bin/env bash
# Lane wg, ON tatooine inside a window (wg-win.sh), tree ~/intarsiaml-wg already built by prep.sh tests:
# the 35B batched step on HIP, three arms in one fresh process each: op listing (which kernels ran), the per-op step profile at B=1/4/16
# (each op synchronised: read the shares) and the unprofiled model-loop step time at B=1/2/4/8/16.
#   branch-knob-off  this branch with the knob unset (registry = main's by construction; the real main binary is a separate gate)
#   opt1             option 1: every kind WMMA at all M, weights row-major (the M=1 W8A8 / bf16 loss is in this number)
#   opt1-frag        option 1 + INTARSIA_HIP_FP8_FRAG=1: dense fp8 weights in fragment order (first real load of the layout)
# (the default table, W8A16 only, is a no-op on the 35B: all its fp8 layers are W8A8; pinned by the wmma_fp8_selection_is_pinned lib test)
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
  echo "  weight layout line: $(grep -h 'relayout:' "$LOG/profile-$name.log" | sort -u | tr '\n' '|')"
}
arm branch-knob-off WG_ARM=knob-off
arm opt1 INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all
arm opt1-frag INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all INTARSIA_HIP_FP8_FRAG=1
echo PROFILE DONE
# tail of window D: the one-instruction hardware probe (raw dump for the CPU model fit); skips itself when not built for this tree
[ -f "$HOME/wg-probe.sh" ] && bash "$HOME/wg-probe.sh" run
