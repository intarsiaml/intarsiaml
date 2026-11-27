#!/usr/bin/env bash
# Window B1 (lane ink-hip-pf2-tiny; radiance down < 30 s): bisect INTARSIA_HIP_INK_PF2 on `inkling2_tiny_g1b_teacher_forced_bf16`,
# main's own test binary (origin/main b7450223e, lane ink-hip-p12's `inkling2-test-main3`, md5 dde0c854), hip, card 0. Masks 151
# minus one bit each (150 / 149 / 147 / 135 / 23), each twice; controls 151 and 0 twice; then each bit alone (1 / 2 / 4 / 16 / 128).
set -uo pipefail
R=$HOME/intarsiaml-ink-hip-p12/docs/results/2026-11-26-inkling2-hip-p12/raw
now() { date '+%T'; }
g1b() {
  local m=$1 rep=$2
  local out; out=$(bash "$R/run.sh" env INTARSIA_INKLING2_BACKEND=hip INTARSIA_HIP_INK_PF2=$m /work/.scratch/bin/inkling2-test-main3 inkling2_tiny_g1b_teacher_forced_bf16 --exact --test-threads 1 2>&1)
  echo "G1B mask=$m r$rep $(echo "$out" | grep -aoE 'test inkling2_tiny_g1b_teacher_forced_bf16 \.\.\. [A-Za-z]+' | awk '{print $NF}') | $(echo "$out" | grep -aE 'rel err' | head -1) [$(now)]"
}
for m in 151 0 150 149 147 135 23; do g1b $m 1; g1b $m 2; done
for m in 1 2 4 16 128; do [ $SECONDS -lt 22 ] && g1b $m 1; done
echo "[stage] done $(now) (${SECONDS}s)"
