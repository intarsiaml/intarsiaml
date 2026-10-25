#!/usr/bin/env bash
# Lane wg, ON tatooine inside a window (wg-win.sh): the B=1 question. Option 1 vs the knob off, interleaved x3 (noise), model-loop step time at
# B = 1,2,4,16, then one traced run per arm (INTARSIA_HIP_TRACE=1: GPU time per kernel name = the per-fold numbers, in situ on the 35B).
set -uo pipefail
export HIP_VISIBLE_DEVICES=0
LOG=$HOME/wg-logs; mkdir -p "$LOG"
DEV="env INTARSIA_REPO=$HOME/intarsiaml-wg INTARSIA_TARGET_VOL=intarsia-cargo-target-wg $HOME/intarsiaml-wg/deploy/tatooine/dev.sh"
T="cargo test --release --locked -p intarsia-models --test batch_cuda_real -- --include-ignored --test-threads=1 --nocapture real_35b_batch_step_timing"
OPT1="INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all"
run() {  # tag, env...
  local tag=$1; shift
  env "$@" $DEV env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=/models/Qwen3.6-35B-A3B-NVFP4 INTARSIA_TIMING_ROWS=${ROWS:-1,2,4,16} $T > "$LOG/folds-$tag.log" 2>&1
  echo "$tag exit=$?  $(grep -E 'TIMING batch B= ?(1|16):' "$LOG/folds-$tag.log" | sed 's/ aggregate.*//' | tr '\n' ' ')"
}
for i in 1 2 3; do
  run off-$i WG_ARM=off
  run opt1-$i $OPT1
done
ROWS=1 run off-trace INTARSIA_HIP_TRACE=1
ROWS=1 run opt1-trace $OPT1 INTARSIA_HIP_TRACE=1
echo FOLDS DONE
