#!/usr/bin/env bash
# Window 7 (lane ink-hip-p12, brief 3; radiance down <= 90 s): attribute `inkling2_tiny_g1b_teacher_forced_bf16` on HIP with main's
# own test binary (origin/main b7450223e, `inkling2-test-main3`). One run each: no knob (control), INTARSIA_HIP_INK_MOE_WMMA=0,
# INTARSIA_HIP_INK_PF2=0, all three of MOE_WMMA=0 ATTN_FLASH=0 PF2=0; then, if time is left, ATTN_FLASH=0 alone and all three plus
# INTARSIA_INK_ROWS_EXACT=0. The test has no chunk knob (it loads with chunk 64 and `ink_dump` runs the 40-token prompt as one pass).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-p12
R=$REPO/docs/results/2026-11-26-inkling2-hip-p12/raw
now() { date '+%T'; }
g1b() {  # label env...
  local lbl=$1; shift
  echo "G1B [$lbl] $(now)"
  bash "$R/run.sh" env INTARSIA_INKLING2_BACKEND=hip "$@" /work/.scratch/bin/inkling2-test-main3 inkling2_tiny_g1b_teacher_forced_bf16 --exact --test-threads 1 2>&1 \
    | grep -aE "^test |test result|rel err" | head -5
}
g1b control X=1
g1b moe_wmma0 INTARSIA_HIP_INK_MOE_WMMA=0
g1b pf2_0 INTARSIA_HIP_INK_PF2=0
g1b all3 INTARSIA_HIP_INK_MOE_WMMA=0 INTARSIA_HIP_INK_ATTN_FLASH=0 INTARSIA_HIP_INK_PF2=0
[ $SECONDS -lt 55 ] && g1b flash0 INTARSIA_HIP_INK_ATTN_FLASH=0
[ $SECONDS -lt 65 ] && g1b all3_rowsexact0 INTARSIA_HIP_INK_MOE_WMMA=0 INTARSIA_HIP_INK_ATTN_FLASH=0 INTARSIA_HIP_INK_PF2=0 INTARSIA_INK_ROWS_EXACT=0
echo "[stage] done $(now) (${SECONDS}s)"
