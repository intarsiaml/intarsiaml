#!/usr/bin/env bash
# Lane cmtp G0 on Metal (endor, this Mac), after scripts/tp2-hip/metal-noop.sh: plan show (text, JSON) + 256 greedy ids, Qwen3-8B and
# Qwen3.6-35B-A3B NVFP4, short and ~3.5k prompts, the lane head against its base (origin/main), flags off.
#   fleetq run --host endor --mem 30 --expect 40 --label "cmtp G0 metal" -- scripts/cmtp/metal-noop.sh
# Trees: the lane worktree and a detached worktree of the base at target/base; build dirs target/metal-{base,dev}.
set -uo pipefail
DEV=$(git rev-parse --show-toplevel)
BASE_REV=$(git -C "$DEV" rev-parse origin/main)
BASE=$DEV/target/base
O=${OUT:-$DEV/docs/results/2026-10-26-cmtp-cuda-mtp-default/raw/metal}; mkdir -p "$O"
M8=$HOME/models/Qwen3-8B; M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4
[[ -d $BASE ]] || git -C "$DEV" worktree add -q --detach "$BASE" "$BASE_REV"
git -C "$BASE" checkout -q --detach "$BASE_REV"
for t in base dev; do
  if [ $t = base ]; then T=$BASE; else T=$DEV; fi
  echo "== build $t $(git -C $T rev-parse --short HEAD) $(date +%T)"
  (cd $T && CARGO_TARGET_DIR=$DEV/target/metal-$t cargo build --release --features metal -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
  RUN="$DEV/target/metal-$t/release/intarsia" bash $DEV/scripts/tp2-hip/noop/noop.sh run $O $t metal $M8 $M35 $DEV/scripts/tp2-hip/noop/prompts
done
bash $DEV/scripts/tp2-hip/noop/noop.sh cmp $O base dev | tee "$O/cmp.txt"
echo "== kernels test metal (dev) $(date +%T)"
$DEV/target/metal-dev/release/intarsia kernels test --backend metal < /dev/null 2>&1 | tail -3
echo "== done $(date +%T)"
