#!/usr/bin/env bash
# Lane wg, ON tatooine inside a window (wg-win.sh): where does option 1's +0.25 ms at B=1 come from? (per-op profile adds only 0.02 ms)
#   Per arm (off / opt1), ROWS=1, 35B real_35b_batch_step_timing:
#     1. timing with INTARSIA_BATCH_GRAPHS=0 (eager B=1 batched step): is the delta still there without the captured graph?
#     2. the same with INTARSIA_HIP_TRACE=1 (per-kernel GPU time of the eager steps): scripts/wgfit/chase_fit.py diffs the two tables.
# Time-boxed: WG_BUDGET seconds (default 330); a step that would start after the budget is skipped. Never fails the window.
set -uo pipefail
export HIP_VISIBLE_DEVICES=0
T0=$(date +%s); BUDGET=${WG_BUDGET:-330}
LOG=$HOME/wg-logs/chase; mkdir -p "$LOG"
REPO=$HOME/intarsiaml-wg
DEV="env INTARSIA_REPO=$REPO INTARSIA_TARGET_VOL=intarsia-cargo-target-wg $REPO/deploy/tatooine/dev.sh"
T="cargo test --release --locked -p intarsia-models --test batch_cuda_real -- --include-ignored --test-threads=1 --nocapture real_35b_batch_step_timing"
OPT1="INTARSIA_HIP_WMMA_ROWS=1 INTARSIA_HIP_WMMA_TABLE=fp8a8=all,bf16=all"
left() { [ $(( $(date +%s) - T0 )) -lt "$BUDGET" ]; }
INNER="env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=/models/Qwen3.6-35B-A3B-NVFP4 INTARSIA_TIMING_ROWS=1"
tm() {  # tag, host env (arm knobs)..., then inner env after "--"
  local tag=$1; shift; left || { echo "chase: skip $tag (budget)"; return; }
  env "$@" $DEV $INNER INTARSIA_BATCH_GRAPHS=0 $T > "$LOG/tm-$tag.log" 2>&1
  echo "$tag exit=$? $(grep -aE 'TIMING (solo|batch B= ?1)' "$LOG/tm-$tag.log" | sed 's/ aggregate.*//' | tr '\n' ' ')"
}
# (rocprofv3 cannot run in the dev image: `env: error while loading shared libraries: libdw.so.1`, window E 2026-10-25.)
# With INTARSIA_BATCH_GRAPHS=0 the B=1 batched steps are eager, so the engine's own INTARSIA_HIP_TRACE (event pair per launch) covers them;
# the solo steps are graph replays and are not traced. Prefill is identical in both arms, so the table diff is the decode difference.
tm off-trace WG_ARM=off INTARSIA_HIP_TRACE=1
tm opt1-trace $OPT1 INTARSIA_HIP_TRACE=1
tm off-nograph-2 WG_ARM=off
tm opt1-nograph-2 $OPT1
echo "CHASE DONE $(( $(date +%s) - T0 )) s"
