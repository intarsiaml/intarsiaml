#!/usr/bin/env bash
# Lane tp2: Metal no-op of the lane head against the lane's base (origin/main 2c2d9941) on endor (after scripts/mx/metal-noop.sh):
# plan show + 256 greedy ids (Qwen3-8B, Qwen3.6-35B-A3B NVFP4; short and ~3.5k prompts), kernels test, workspace tests on the head.
# Runs ON endor inside a fleetq job (--exclusive). Worktrees /opt/code/wt-tp2-noop-base (2c2d9941) and the lane head's own worktree.
set -uo pipefail
BASE=${BASE:-/opt/code/wt-tp2-noop-base}; DEV=${DEV:-/opt/code/wt-tp2-batch}
O=${OUT:-$DEV/docs/results/2026-10-25-tp2-hip/raw/metal}; mkdir -p "$O"
M8=$HOME/models/Qwen3-8B; M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4
TG=/opt/code/intarsiaml/.worktrees/tp2-targets; mkdir -p $TG
for t in base dev; do
  [ -d $TG/$t-metal ] || cp -cR /opt/code/intarsiaml/.worktrees/base-metal $TG/$t-metal 2>/dev/null || true
  if [ $t = base ]; then T=$BASE; else T=$DEV; fi
  echo "== build $t $(git -C $T rev-parse --short HEAD) $(date +%T)"
  (cd $T && CARGO_TARGET_DIR=$TG/$t-metal cargo build --release --features metal -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
  RUN="$TG/$t-metal/release/intarsia" bash $DEV/scripts/tp2-hip/noop/noop.sh run $O $t metal $M8 $M35 $DEV/scripts/tp2-hip/noop/prompts
done
bash $DEV/scripts/tp2-hip/noop/noop.sh cmp $O base dev
echo "== kernels test metal (dev) $(date +%T)"
$TG/dev-metal/release/intarsia kernels test --backend metal < /dev/null 2>&1 | tail -3
[ -d $TG/test ] || cp -cR /opt/code/intarsiaml/.worktrees/test $TG/test 2>/dev/null || true
echo "== workspace tests on the head $(date +%T)"
(cd $DEV && CARGO_TARGET_DIR=$TG/test cargo test --workspace --release --no-fail-fast 2>&1 | grep -E "^test result|FAILED|failed|panicked" | sort | uniq -c | sort -rn | head -20)
echo "== done $(date +%T)"
