#!/bin/zsh
# Lane d8: Metal no-op gate on endor (the loader and qwen3.toml are shared files). Base = origin/main 602a1fcfe, branch = this tree.
#   scripts/d8/metal_noop.sh build   (fleetq run --host endor --load cpu --expect 60 ...)   two release builds under .scratch/
#   scripts/d8/metal_noop.sh run     (fleetq run --host endor --exclusive --expect 25 ...)  plan show, 256 greedy ids, kernels test: base vs branch
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$REPO"
S=$REPO/.scratch; M=/Users/josef/models/Qwen3-8B; BASEREV=${BASEREV:-602a1fcfe}
OUT=$REPO/docs/results/2026-11-05-dflash-8b-default/raw/metal; mkdir -p "$OUT" "$S"
case "${1:?}" in
build)
  rm -rf "$S/base-src"; mkdir -p "$S/base-src"; git archive "$BASEREV" | tar -x -C "$S/base-src"
  ( cd "$S/base-src" && CARGO_TARGET_DIR=$S/target-base cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -2 ) && cp "$S/target-base/release/intarsia" "$S/intarsia.base"
  ( cd "$REPO" && CARGO_TARGET_DIR=$S/target-branch cargo build --release --locked -p intarsia-cli --features metal 2>&1 | tail -2 ) && cp "$S/target-branch/release/intarsia" "$S/intarsia.branch"
  shasum -a 256 "$S/intarsia.base" "$S/intarsia.branch" ;;
run)
  fails=0
  for t in base branch; do
    B=$S/intarsia.$t
    $B plan show --backend metal --model $M --max-seq 8192 < /dev/null > $OUT/plan-$t.txt 2> /dev/null
    $B plan show --backend metal --model $M --max-seq 8192 --json < /dev/null > $OUT/plan-$t.json 2> /dev/null
    for p in short 4k; do
      $B generate -m $M --backend metal --max-seq 8192 --chat --no-think --temperature 0 --profile-every 0 --ignore-eos --show-ids \
        -p "$(cat scripts/hg/prompts/$p.txt)" --max-tokens 256 < /dev/null > /dev/null 2> $OUT/ids-$p-$t.err
      grep -a '^ids:' $OUT/ids-$p-$t.err > $OUT/ids-$p-$t.ids
    done
    $B kernels test --backend metal < /dev/null > $OUT/kernels-$t.txt 2>&1
  done
  for k in txt json; do cmp -s $OUT/plan-base.$k $OUT/plan-branch.$k && [ -s $OUT/plan-branch.$k ] && echo "PASS plan show $k unchanged ($(wc -c < $OUT/plan-branch.$k) B)" || { echo "FAIL plan show $k"; fails=$((fails+1)); }; done
  for p in short 4k; do cmp -s $OUT/ids-$p-base.ids $OUT/ids-$p-branch.ids && [ -s $OUT/ids-$p-branch.ids ] && echo "PASS 256 greedy ids $p identical ($(tr ',' '\n' < $OUT/ids-$p-branch.ids | wc -l) ids)" || { echo "FAIL ids $p"; fails=$((fails+1)); }; done
  for t in base branch; do echo "kernels $t: $(grep -a -c ADMITTED $OUT/kernels-$t.txt) ADMITTED rows, $(grep -a -ci 'fail' $OUT/kernels-$t.txt) lines with fail; tail: $(tail -1 $OUT/kernels-$t.txt | cut -c1-160)"; done
  echo "metal no-op fails=$fails" ;;
esac
