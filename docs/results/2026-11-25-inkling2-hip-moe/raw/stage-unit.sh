#!/usr/bin/env bash
# Unit stage of lane ink-hip-moe (inside window.sh): correctness vs the cpu reference, rows-exactness, then the microbenchmark.
set -uo pipefail
cd "$HOME/intarsiaml-ink-hip-moe"
R="bash .scratch/run.sh"
rc=0
$R env INTARSIA_REQUIRE_GPU=1 ${INK_MOE_BIG:+INK_MOE_BIG=1} cargo test --release -j8 -p intarsia-backend-hip --test ink_moe -- --nocapture --test-threads=1 ink_moe_matches_cpu_reference ink_moe_rows_exact || rc=$?
echo "[stage] tests exit $rc"
if [ "${INK_BENCH:-1}" = 1 ]; then
  $R env INTARSIA_REQUIRE_GPU=1 INK_MOE_BENCH=1 ${INK_MOE_BENCH_T:+INK_MOE_BENCH_T=$INK_MOE_BENCH_T} cargo test --release -j8 -p intarsia-backend-hip --test ink_moe -- --nocapture --test-threads=1 ink_moe_bench || rc=$?
  echo "[stage] bench exit $rc"
fi
exit $rc
