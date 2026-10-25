#!/usr/bin/env bash
# Lane l7 window 1, ON tatooine inside a fleetq slot that takes radiance down: the moe_rows_timing microbench (card 0). Needs build.sh dev.
# trap restores radiance. Log: ~/intarsiaml-l7-logs/w1-timing.log
set -uo pipefail
S=$HOME/intarsiaml-l7-hip/scripts/l7
O=$HOME/intarsiaml-l7-logs; mkdir -p "$O"
D=$HOME/intarsiaml-l7-hip/deploy/tatooine/dev.sh
trap '$S/win.sh restore' EXIT
$S/win.sh stop
export INTARSIA_REGISTRY_VOL=intarsia-cargo-l7-registry
echo "[w1] $(date +%T) timing"
INTARSIA_REPO=$HOME/intarsiaml-l7-hip INTARSIA_TARGET_VOL=intarsia-cargo-l7-dev HIP_VISIBLE_DEVICES=0 \
  INTARSIA_HIP_MOE_ROWS_BENCH=1 INTARSIA_HIP_MOE_ROWS_BENCH_T=${W1_T:-5,8,12,16} INTARSIA_REQUIRE_GPU=1 \
  $D cargo test --release -p intarsia-backend-hip --test moe_block moe_rows_timing -- --nocapture --test-threads=1 < /dev/null 2>&1 \
  | grep -aE "^moe_rows_timing|^  \(|^test |panicked|^error" | tee "$O/w1-timing.log"
echo "[w1] done $(date +%T)"
