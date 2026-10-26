#!/usr/bin/env bash
# Lane l7, ON tatooine inside a fleetq slot that takes radiance down (card 0 only): the correctness gates. Needs build.sh. Logs: ~/intarsiaml-l7-logs/gates.
# gates.sh [sections: "GB GC NOOP"]; trap restores radiance. Keep one window <= 30 min: run the sections in separate windows if needed.
#   GB   : G-B (tests/moe_block.rs incl. the grouped block, negative controls, graph replay; rows_exact; rows_registry)
#   GC   : G-C 35B batched == solo (17,550 pairs) + batch_recurrent (tiny hybrid)
#   BENCH: moe_rows_timing at T = 3, 4 (is the grouped block better than the fused rows block there?)
#   NOOP : batching-off no-op, base vs dev: plan show (text + json) and 256 greedy ids (8B and 35B, short and 4k prompt)
set -uo pipefail
SECTIONS=${*:-GB GC NOOP}
S=$HOME/intarsiaml-l7-hip/scripts/l7
O=$HOME/intarsiaml-l7-logs/gates; mkdir -p "$O"
D=$HOME/intarsiaml-l7-hip/deploy/tatooine/dev.sh
RD=$HOME/intarsiaml-l7-hip; RB=$HOME/intarsiaml-l7-base
VD=intarsia-cargo-l7-dev; VB=intarsia-cargo-l7-base
M8=/models/Qwen3-8B; M35=/models/Qwen3.6-35B-A3B-NVFP4
export INTARSIA_REGISTRY_VOL=intarsia-cargo-l7-registry
want() { [[ " $SECTIONS " == *" $1 "* ]]; }
trap '$S/win.sh restore' EXIT
$S/win.sh stop
dev() { # dev <repo> <vol> <cmd...>
  r=$1; v=$2; shift 2
  INTARSIA_REPO=$r INTARSIA_TARGET_VOL=$v HIP_VISIBLE_DEVICES=0 $D "$@" < /dev/null
}
F='^test |^test result|FAILED|panicked|^error|pairs|bitwise|launches|negative|SKIPPED'
if want GB; then
  echo "[gates] GB $(date +%T)"
  dev $RD $VD env INTARSIA_REQUIRE_GPU=1 cargo test --release --no-fail-fast -p intarsia-backend-hip --test moe_block --test rows_exact --test rows_registry -- --test-threads=1 --nocapture 2>&1 | grep -aE "$F|moe_block_grouped" | tee "$O/g-b.log"
fi
if want GC; then
  echo "[gates] GC $(date +%T)"
  dev $RD $VD env INTARSIA_BATCH_TEST_BACKEND=hip INTARSIA_REQUIRE_GPU=1 INTARSIA_QWEN36_DIR=$M35 cargo test --release --no-fail-fast -p intarsia-models --test batch_cuda_real --test batch_recurrent -- --test-threads=1 --nocapture 2>&1 | grep -aE "$F" | tee "$O/g-c.log"
fi
if want BENCH; then
  echo "[gates] BENCH $(date +%T)"
  dev $RD $VD env INTARSIA_HIP_MOE_ROWS_BENCH=1 INTARSIA_HIP_MOE_ROWS_BENCH_T=${BENCH_T:-3,4} INTARSIA_HIP_MOE_ROWS_BENCH_G2=2:2 INTARSIA_REQUIRE_GPU=1 \
    cargo test --release -p intarsia-backend-hip --test moe_block moe_rows_timing -- --nocapture --test-threads=1 2>&1 | grep -aE "^moe_rows_timing|^  \(|^  G2=|^test |panicked|^error" | tee "$O/bench.log"
fi
if want NOOP; then
  echo "[gates] NOOP $(date +%T)"
  for t in base dev; do
    if [ $t = base ]; then r=$RB; v=$VB; else r=$RD; v=$VD; fi
    RUN="env INTARSIA_REPO=$r INTARSIA_TARGET_VOL=$v HIP_VISIBLE_DEVICES=0 $D /target/release/intarsia" \
      bash $RD/scripts/hg/noop.sh run "$O" $t hip $M8 $M35 $RD/scripts/hg/prompts < /dev/null 2>&1 | tee -a "$O/noop.log"
  done
  bash $RD/scripts/hg/noop.sh cmp "$O" base dev | tee -a "$O/noop.log"
fi
echo "[gates] done $(date +%T)"
