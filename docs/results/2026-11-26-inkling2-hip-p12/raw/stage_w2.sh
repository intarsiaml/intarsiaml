#!/usr/bin/env bash
# Window 2 (lane ink-hip-p12): gate 1 again on p12b (the new launch-shape defaults), the microbench at 1 / 2 / 4 / 8 rows (the new
# defaults against bf16), then stage_gate.sh: md5 at 64 tokens (spec off and default, main / branch, branch at PACK12=0), G2 short /
# long (main and branch), admission (matmul_p12, ink_in_proj), the tiny suites on hip.
set -uo pipefail
REPO=$HOME/intarsiaml-ink-hip-p12
R=$REPO/docs/results/2026-11-26-inkling2-hip-p12/raw
now() { date '+%F %T'; }
echo "[stage] bitwise tests $(now)"
bash "$R/run.sh" env INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test ink_p12 -- --nocapture --test-threads=1 2>&1 | grep -aE "^test |test result|bitwise|escapes|panicked|differs|error" | head -20
echo "[stage] microbench rows 1 2 4 8 $(now)"
bash "$R/run.sh" env P12_BENCH=1 P12_MS=1,2,4,8 INTARSIA_REQUIRE_GPU=1 cargo test --release -p intarsia-backend-hip --test ink_p12 p12_microbench -- --nocapture --test-threads=1 2>&1 | grep -aE "^BENCH|test result|panicked|error" | head -80
BR=${BR:-p12b} PARTS="md5 g2 adm tiny" bash "$R/stage_gate.sh"
