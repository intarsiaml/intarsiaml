#!/usr/bin/env bash
# Window 1 (lane ink-hip-i8), kernel-only: the i8 correctness tests (real shapes + small / tails + every launch shape) first (<= 60 s),
# then the step-2 microbench: int8 per-row vs main's P12 kernels at the real shapes (wo, unembed, in_proj) at rows I8_MS (default 1,4).
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-i8
R=$REPO/docs/results/2026-11-26-inkling2-hip-i8/raw
now() { date '+%F %T'; }
echo "[stage] i8 correctness $(now)"
bash "$R/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test ink_i8 -- --nocapture --test-threads=1 2>&1 | grep -aE "^test |test result|checked|panicked|differs|error|got " | head -30
echo "[stage] microbench $(now)"
bash "$R/run.sh" env I8_BENCH=1 I8_MS=${I8_MS:-1,4} INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test ink_i8 i8_microbench -- --nocapture --test-threads=1 2>&1 | grep -aE "^BENCH|test result|panicked|error" | head -120
echo "[stage] done $(now)"
