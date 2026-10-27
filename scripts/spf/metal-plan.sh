#!/usr/bin/env bash
# Lane spf: Metal `plan show` (json + text) of origin/main against this tree, 8B and 35B NVFP4, byte-compared (this change touches kernels/cuda only).
#   fleetq run --host endor --load cpu --mem 20 --expect 25 --label "spf: metal plan show" -- scripts/spf/metal-plan.sh
set -uo pipefail
DEV=$(git rev-parse --show-toplevel); BASE=$DEV/target/base
O=${OUT:-$DEV/docs/results/2026-10-27-spf-cuda-prefill/raw/metal}; mkdir -p "$O"
M8=$HOME/models/Qwen3-8B; M35=$HOME/models/Qwen3.6-35B-A3B-NVFP4
[[ -d $BASE ]] || git -C "$DEV" worktree add -q --detach "$BASE" origin/main
git -C "$BASE" checkout -q --detach origin/main
fails=0
for t in base dev; do
  if [ $t = base ]; then T=$BASE; else T=$DEV; fi
  (cd $T && CARGO_TARGET_DIR=$DEV/target/metal-$t cargo build --release --features metal -p intarsia-cli 2>&1 | grep -E '^error|Finished' | tail -2)
  for mm in 8:$M8 35:$M35; do
    m=${mm%%:*}; M=${mm#*:}
    $DEV/target/metal-$t/release/intarsia plan show --backend metal --model $M --max-seq 8192 --json < /dev/null > $O/plan-$m-$t.json 2> /dev/null
    $DEV/target/metal-$t/release/intarsia plan show --backend metal --model $M --max-seq 8192 < /dev/null > $O/plan-$m-$t.txt 2> /dev/null
  done
done
for m in 8 35; do for k in json txt; do
  if cmp -s $O/plan-$m-base.$k $O/plan-$m-dev.$k && [ -s $O/plan-$m-base.$k ]; then echo "PASS metal plan $m $k byte-identical ($(wc -c < $O/plan-$m-dev.$k) B)"
  else echo "FAIL metal plan $m $k"; fails=$((fails+1)); fi
done; done
echo "metal plan fails=$fails"
