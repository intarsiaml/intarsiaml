#!/usr/bin/env bash
# Lane wg, ON endor inside `fleetq run --host endor --exclusive --mem 24`: Metal no-op of the merged branch head against main 0c142a33
# (batching off, knob unset): plan show text+json and 256 greedy ids after a short and a ~3.5k-token prompt (8B, 35B), kernels test, workspace tests.
# BASE and DEV are checked-out trees (git worktrees under the project's .worktrees/); each builds in its own target dir.
set -uo pipefail
DEV=${DEV:-/opt/code/wt-wg}; BASE=${BASE:-/opt/code/wt-wg/.worktrees/base}; TG=${TG:-/opt/code/wt-wg/.worktrees/targets}
O=${OUT:-$DEV/docs/results/2026-10-25-wg-hip-wmma-gemv/raw/merge-gates/metal}; mkdir -p "$O" "$TG"
M8=$HOME/models/Qwen3-8B; M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4
now() { date '+%F %T'; }
echo "[metal] start $(now) base $(git -C $BASE rev-parse --short HEAD) dev $(git -C $DEV rev-parse --short HEAD)"
for t in base dev; do
  if [ $t = base ]; then T=$BASE; else T=$DEV; fi
  echo "== build $t $(now)"
  (cd $T && CARGO_TARGET_DIR=$TG/$t-metal cargo build --release --features metal -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
  [ -x $TG/$t-metal/release/intarsia ] || { echo "BUILD FAILED $t"; exit 1; }
done
for t in base dev; do
  RUN="$TG/$t-metal/release/intarsia" bash $DEV/scripts/hg/noop.sh run $O $t metal $M8 $M35 $DEV/scripts/hg/prompts
done
bash $DEV/scripts/hg/noop.sh cmp $O base dev | tee "$O/cmp.txt"
echo "== kernels test metal (dev) $(now)"
$TG/dev-metal/release/intarsia kernels test --backend metal < /dev/null 2>&1 | tail -3
echo "== workspace tests on the branch (--features metal) $(now)"
(cd $DEV && CARGO_TARGET_DIR=$TG/test cargo test --workspace --features metal --release --no-fail-fast 2>&1 | grep -E "^test result|FAILED|failed|panicked|^error" | sort | uniq -c | sort -rn | head -30)
echo "== done $(now)"
