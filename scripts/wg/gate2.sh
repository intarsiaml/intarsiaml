#!/usr/bin/env bash
# Lane wg, ON tatooine inside a window: wmma_folds only (fold vs chain), three activation forms, on the tree already built in ~/intarsiaml-wg.
set -uo pipefail
export HIP_VISIBLE_DEVICES=0
LOG=$HOME/wg-logs; mkdir -p "$LOG"
DEV="env INTARSIA_REPO=$HOME/intarsiaml-wg INTARSIA_TARGET_VOL=intarsia-cargo-target-wg $HOME/intarsiaml-wg/deploy/tatooine/dev.sh"
for dm in 4 1 65; do
  echo "=== wmma_folds DIRECT_MIN=$dm $(date +%T)"
  env INTARSIA_HIP_WMMA_DIRECT_MIN=$dm INTARSIA_REQUIRE_GPU=1 INTARSIA_HIP_DEVICE=0 $DEV cargo test --release --locked -p intarsia-backend-hip --test wmma_folds --no-fail-fast -- --test-threads=1 > "$LOG/folds-dm$dm.log" 2>&1
  echo "exit=$?  $(grep -E '^test result' "$LOG/folds-dm$dm.log" | tr '\n' ' ')"
  grep -E "^test .*(FAILED|ok)$|panicked|differs|SKIPPED|rror" "$LOG/folds-dm$dm.log" | head -30
done
echo GATE2 DONE
