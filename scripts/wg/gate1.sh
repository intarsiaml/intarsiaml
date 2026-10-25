#!/usr/bin/env bash
# Lane wg, ON tatooine inside a window (wg-win.sh): the WMMA correctness gate on the tree in ~/intarsiaml-wg (already built by prep.sh tests):
# rows_exact (wmma_*) and wmma_folds under three activation forms (default, all-direct, all-staged), then the negative controls (must FAIL).
set -uo pipefail
export HIP_VISIBLE_DEVICES=0
LOG=$HOME/wg-logs; mkdir -p "$LOG"
DEV="env INTARSIA_REPO=$HOME/intarsiaml-wg INTARSIA_TARGET_VOL=intarsia-cargo-target-wg $HOME/intarsiaml-wg/deploy/tatooine/dev.sh"
T="cargo test --release --locked -p intarsia-backend-hip --test rows_exact --test wmma_folds --no-fail-fast"
run() {  # name, env assignments..., then filter
  local name=$1 filter=$2; shift 2
  echo "=== $name  ($*)  filter=$filter  $(date +%T)"
  env "$@" INTARSIA_REQUIRE_GPU=1 INTARSIA_HIP_DEVICE=0 $DEV $T -- --test-threads=1 "$filter" > "$LOG/$name.log" 2>&1
  echo "exit=$?  $(grep -E '^test result' "$LOG/$name.log" | tr '\n' ' ')"
  grep -E "^test .*(FAILED|ok)$|panicked|differs|SKIPPED" "$LOG/$name.log" | head -40
}
run gate-default wmma INTARSIA_HIP_WMMA_DIRECT_MIN=4
run gate-alldirect wmma INTARSIA_HIP_WMMA_DIRECT_MIN=1
run gate-allstaged wmma INTARSIA_HIP_WMMA_DIRECT_MIN=65
echo "--- negative controls (each must FAIL)"
run ctl-fp8-rows-as-mma_f8 wmma_fp8_rows_are_bitwise INTARSIA_HIP_GB_CONTROL=matmul_fp8_wmma_rows=matmul_fp8_mma_f8
run ctl-bf16-rows-as-matmul_bf16 wmma_bf16_rows_are_bitwise INTARSIA_HIP_GB_CONTROL=matmul_bf16_wmma_rows=matmul_bf16
echo GATE1 DONE
