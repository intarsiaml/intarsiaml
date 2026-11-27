#!/usr/bin/env bash
# Window 6 (lane ink-hip-i8), card 0: the hip tiny suites `inkling2` and `inkling2_batch` (INTARSIA_INKLING2_BACKEND=hip) with the i8 env
# (INTARSIA_INK_ATTN_QUANT=int8_row INTARSIA_LMHEAD_QUANT=int8_row), then the same without it (the reference list of by-design failures).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-i8
R=$REPO/docs/results/2026-11-26-inkling2-hip-i8/raw
OUT=$REPO/logs/w6
mkdir -p "$OUT"
now() { date '+%T'; }
t() {  # name test env...
  local n=$1 tst=$2; shift 2
  echo "[stage] $n $(now)"
  bash "$R/run.sh" env INTARSIA_INKLING2_BACKEND=hip "$@" cargo test --release -p intarsia-models --test "$tst" -- --test-threads 1 > "$OUT/$n.log" 2>&1
  echo "TEST $n exit $? | $(grep -aE '^test result' "$OUT/$n.log" | tr '\n' ' ') | FAILED: $(grep -aE '^test .* FAILED' "$OUT/$n.log" | sed -e 's/^test //' -e 's/ \.\.\. FAILED//' | tr '\n' ' ')"
}
I8="INTARSIA_INK_ATTN_QUANT=int8_row INTARSIA_LMHEAD_QUANT=int8_row"
t inkling2-i8 inkling2 $I8
t inkling2_batch-i8 inkling2_batch $I8
t inkling2-main inkling2
t inkling2_batch-main inkling2_batch
echo "[stage] done $(now)"
