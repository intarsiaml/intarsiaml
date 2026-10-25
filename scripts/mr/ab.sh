#!/usr/bin/env bash
# Lane mr, LOCAL (endor), with the fleetq job variable set in the environment to the id of the slot that holds tatooine with
# radiance down (localeval then runs inside that slot instead of queueing):
# 27B MXFP4 batched decode A/B, base (origin/main) vs dev, `serve --max-batch 1` (main's only 27B mode), `8` and `16`, localeval speed at 1/4/8/16 streams.
#   <fleetq job env>=<id> scripts/mr/ab.sh [arms: "base:8 dev:8 base:16 dev:16"] [tag]
set -uo pipefail
arms=${1:-"base:1 dev:1 base:8 dev:8 dev:16"}; tag=${2:-mr}
LOG=${LOG:-/tmp/mr-ab}; mkdir -p "$LOG"
for a in $arms; do
  tree=${a%%:*}; mb=${a##*:}
  echo "[ab $tree mb=$mb] up $(date +%T)"
  ssh tatooine "bash ~/mr-hip/scripts/mr/serve.sh up $tree $mb" 2>&1 | tail -4 | tee "$LOG/$tree-$mb.up"
  if grep -q "serving after" "$LOG/$tree-$mb.up"; then
    conc=4,8,16  # streams above max-batch queue; the single-stream cell is the sweep itself
    (cd /opt/code/deepeval && timeout 1500 uv run localeval speed --url http://tatooine:8294/v1 --model intarsia --sizes 1024 --gen 128 --reps 3 \
      --concurrency $conc --label "$tag-$tree-mb$mb" --no-unim \
      --recipe "intarsia serve --backend hip --model Qwen3.8-27B-Quark-AWQ-MXFP4 --max-seq 4096 --max-batch $mb tree=$tree") > "$LOG/$tree-$mb.out" 2>&1
    echo "[ab $tree mb=$mb] localeval rc=$?"; tail -12 "$LOG/$tree-$mb.out" | cut -c1-200
  fi
  ssh tatooine "bash ~/mr-hip/scripts/mr/serve.sh down $tag-$tree-mb$mb" 2>&1 | tail -1
done
