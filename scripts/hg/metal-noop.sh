#!/usr/bin/env bash
# Lane hg: Metal no-op (batching off) of the branch against origin/main on endor, kernels test, workspace tests on the branch.
# Runs ON endor inside a fleetq job (--exclusive). BASE and DEV are checked-out trees; each builds in its own target dir.
set -uo pipefail
BASE=${BASE:-/opt/code/wt-hg-base}; DEV=${DEV:-/opt/code/wt-hg}
O=${OUT:-$DEV/docs/results/2026-10-25-hg-hip-serve-gap/raw/metal}; mkdir -p "$O"
M8=$HOME/models/Qwen3-8B; M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4
for t in base dev; do
  if [ $t = base ]; then T=$BASE; else T=$DEV; fi
  echo "== build $t"
  (cd $T && CARGO_TARGET_DIR=/opt/code/hg-targets/$t-metal cargo build --release --features metal -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
  RUN="/opt/code/hg-targets/$t-metal/release/intarsia" bash $DEV/scripts/hg/noop.sh run $O $t metal $M8 $M35 $DEV/scripts/hg/prompts
done
bash $DEV/scripts/hg/noop.sh cmp $O base dev
echo "== kernels test metal (dev)"
/opt/code/hg-targets/dev-metal/release/intarsia kernels test --backend metal < /dev/null 2>&1 | tail -3
echo "== workspace tests on the branch"
(cd $DEV && CARGO_TARGET_DIR=/opt/code/hg-targets/test cargo test --workspace --release --no-fail-fast 2>&1 | grep -E "^test result|FAILED|failed|panicked" | sort | uniq -c | sort -rn | head -20)
echo "== done"
