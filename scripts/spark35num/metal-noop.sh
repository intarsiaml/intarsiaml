#!/usr/bin/env bash
# Lane spark35 on Metal (endor, this Mac), the cmtp G0 recipe: plan show (text, JSON) + 256 greedy ids, Qwen3-8B and Qwen3.6-35B-A3B NVFP4, short and ~3.5k prompts,
# the lane head against its base (origin/main), env vars unset: the shared-file change (Backend::gdn_state_supported, Model::gdn_state_dt) must be a no-op on Metal.
#   fleetq run --host endor --mem 30 --expect 40 --label "spark35 metal no-op" -- scripts/spark35num/metal-noop.sh
# Trees: the lane worktree and an export of the base at .scratch/base-src; build dirs target/metal-{base,dev}.
set -uo pipefail
DEV=$(git rev-parse --show-toplevel)
BASE_REV=$(git -C "$DEV" rev-parse origin/main)
BASE=$DEV/.scratch/base-src
O=${OUT:-$DEV/docs/results/2026-11-05-spark35-numerics/raw/metal}; mkdir -p "$O"
M8=$HOME/models/Qwen3-8B; M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4
# the base is a plain export of origin/main (no extra git worktree): lane folders stay inside .scratch/
rm -rf "$BASE"; mkdir -p "$BASE"; git -C "$DEV" archive "$BASE_REV" | tar -x -C "$BASE"
for t in base dev; do
  if [ $t = base ]; then T=$BASE; else T=$DEV; fi
  echo "== build $t $([ $t = base ] && echo ${BASE_REV:0:9} || git -C $T rev-parse --short HEAD) $(date +%T)"
  (cd $T && CARGO_TARGET_DIR=$DEV/target/metal-$t cargo build --release --features metal -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -3)
  RUN="$DEV/target/metal-$t/release/intarsia" bash $DEV/scripts/tp2-hip/noop/noop.sh run $O $t metal $M8 $M35 $DEV/scripts/tp2-hip/noop/prompts
done
bash $DEV/scripts/tp2-hip/noop/noop.sh cmp $O base dev | tee "$O/cmp.txt"
echo "== kernels test metal (dev) $(date +%T)"
$DEV/target/metal-dev/release/intarsia kernels test --backend metal < /dev/null 2>&1 | tail -3
echo "== done $(date +%T)"
