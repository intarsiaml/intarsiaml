#!/usr/bin/env bash
# Lane mxr (from scripts/mx/metal-noop.sh): Metal G0 of the branch against origin/main on endor, inside a fleetq job: plan show +
# 256 greedy ids (Qwen3-8B, Qwen3.6-35B-A3B NVFP4; short and 4k) via scripts/hg/noop.sh, plan show of the 27B MXFP4, and
# `kernels test` on the branch. Trees: .worktrees/mxr (dev) and .worktrees/mxr/.base-metal (origin/main, git worktree);
# target dirs .worktrees/mxr/.targets/{base,dev}-metal.
set -uo pipefail
DEV=/opt/code/intarsiaml/.worktrees/mxr; BASE=$DEV/.base-metal; TG=$DEV/.targets
O=$DEV/docs/results/2026-10-26-mxr-hip-mxfp4-rows/raw/metal; mkdir -p "$O" "$TG"
M8=$HOME/models/Qwen3-8B; M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4; M27=$HOME/models/Qwen3.8-27B-Quark-AWQ-MXFP4
for t in base dev; do
  if [ $t = base ]; then T=$BASE; else T=$DEV; fi
  echo "== build $t $(git -C $T rev-parse --short HEAD) $(date +%T)"
  (cd $T && CARGO_TARGET_DIR=$TG/$t-metal cargo build --release --locked --features metal -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
  R=$TG/$t-metal/release/intarsia
  RUN="$R" bash $DEV/scripts/hg/noop.sh run $O $t metal $M8 $M35 $DEV/scripts/hg/prompts
  $R plan show --backend metal --model "$M27" --max-seq 8192 --json < /dev/null > "$O/plan-27-$t.json" 2> "$O/plan-27-$t.json.err"
  echo "plan 27 $t exit $?"
done
bash $DEV/scripts/hg/noop.sh cmp $O base dev
if cmp -s "$O/plan-27-base.json" "$O/plan-27-dev.json"; then echo "plan 27 json: identical ($(wc -c < "$O/plan-27-dev.json" | tr -d ' ') B)"; else echo "plan 27 json: DIFFER"; fi
echo "== kernels test metal (dev) $(date +%T)"
$TG/dev-metal/release/intarsia kernels test --backend metal < /dev/null 2>&1 | tail -3
echo "== done $(date +%T)"
