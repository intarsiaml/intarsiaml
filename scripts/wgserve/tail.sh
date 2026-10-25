#!/usr/bin/env bash
# Lane wg, ON tatooine (copy: ~/wg-tail.sh), run by served.sh inside the radiance-down window after the last served visit (server stopped, GPU idle).
#   1. the fp8 WMMA ablation bench at M=1 (ABL=1: where the M=1 time goes; needs the IK_F8R_ABLATE build in ~/wg-build-abl, made by `tail.sh build`)
#   2. the model-loop step time at B = 1, 4, 16 for four tables, interleaved x2:
#        off   no knob (the GEMV family, = main)           opt1  fp8a8=all,bf16=all (option 1)
#        fp8   fp8a8=all,bf16=off (WMMA only for W8A8)      bf16  fp8a8=off,bf16=all (WMMA only for bf16)
#      fp8 and bf16 each keep batched == solo bitwise (one kernel per kind at every M): option (d), the per-kind split.
#   tail.sh build          cpu job / pre-step (radiance up): compile the ablation bench
#   tail.sh run [SECS]     the tail, stops starting new steps after SECS (default 270)
# Output: ~/wg-logs/tail/{abl.log,tm-<arm>-<rep>.log}
set -uo pipefail
export HIP_VISIBLE_DEVICES=0
LOG=$HOME/wg-logs/tail; mkdir -p "$LOG"
if [ "${1:-}" = build ]; then
  XFLAGS=-DIK_F8R_ABLATE BUILD=$HOME/wg-build-abl REPO=$HOME/intarsiaml-wg bash "$HOME/wg-bench.sh" build > "$LOG/build-abl.log" 2>&1
  echo "ablation build exit=$? $(tail -n 1 "$LOG/build-abl.log")"; exit 0
fi
T0=$(date +%s); BUDGET=${2:-270}
left() { [ $(( $(date +%s) - T0 )) -lt "$BUDGET" ]; }
REPO=$HOME/intarsiaml-wg
DEV="env INTARSIA_REPO=$REPO INTARSIA_TARGET_VOL=intarsia-cargo-target-wg $REPO/deploy/tatooine/dev.sh"
T="cargo test --release --locked -p intarsia-models --test batch_cuda_real -- --include-ignored --test-threads=1 --nocapture real_35b_batch_step_timing"
INNER="env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=/models/Qwen3.6-35B-A3B-NVFP4 INTARSIA_TIMING_ROWS=1,4,16"
K="INTARSIA_HIP_WMMA_ROWS=1"
if [ -x "$HOME/wg-build-abl/bench" ]; then
  ABL=1 BUILD=$HOME/wg-build-abl REPO=$REPO timeout 100 bash "$HOME/wg-bench.sh" run > "$LOG/abl.log" 2>&1
  echo "abl exit=$? $(grep -c '^abl' "$LOG/abl.log") lines"
else echo "abl: no ablation binary (run tail.sh build first)"; fi
arm_env() { case $1 in off) echo "INTARSIA_WG_ARM=off" ;; opt1) echo "$K INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all" ;;
            fp8) echo "$K INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=off" ;; bf16) echo "$K INTARSIA_HIP_WMMA_TABLE=fp8a8=off,bf16=all" ;; esac; }
for rep in 1 2; do
  for arm in off opt1 fp8 bf16; do
    left || { echo "tail: skip $arm-$rep (budget)"; continue; }
    env $(arm_env $arm) $DEV $INNER $T > "$LOG/tm-$arm-$rep.log" 2>&1
    echo "$arm-$rep exit=$? $(grep -aE 'TIMING batch B= ?(1|4|16):' "$LOG/tm-$arm-$rep.log" | sed -E 's/TIMING batch B= ?//; s/ aggregate.*//' | tr '\n' ' ')"
  done
done
echo "TAIL DONE $(( $(date +%s) - T0 )) s"
