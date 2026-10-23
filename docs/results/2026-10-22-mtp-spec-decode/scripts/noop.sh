#!/usr/bin/env bash
# Runs ON spark1 inside a fleetq slot: Qwen3-8B (and 35B) no-op of the branch vs main, both --spec off (the default):
# `plan show` JSON + text identical, 256 greedy ids identical. Usage: noop.sh <main bin> <branch bin> <out dir>
set -uo pipefail
MAIN=${1:?}; BR=${2:?}; OUT=${3:?}; mkdir -p "$OUT"
M8=$(ls -d $HOME/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1)
M35=$(ls -d $HOME/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
for m in 8b 35b; do
  [[ $m == 8b ]] && M=$M8 || M=$M35
  for side in main br; do
    [[ $side == main ]] && B=$MAIN || B=$BR
    "$B" plan show --backend cuda --model "$M" --max-seq 8192 --mem-budget 30 --json > "$OUT/plan-$m-$side.json" 2> "$OUT/plan-$m-$side.err" < /dev/null
    "$B" plan show --backend cuda --model "$M" --max-seq 8192 --mem-budget 30 > "$OUT/plan-$m-$side.txt" 2>> "$OUT/plan-$m-$side.err" < /dev/null
    "$B" generate --backend cuda --model "$M" -p "The history of the printing press begins" --max-tokens 256 --ignore-eos --show-ids \
      --max-seq 8192 --mem-budget 30 --profile-every 0 > "$OUT/gen-$m-$side.out" 2> "$OUT/gen-$m-$side.err" < /dev/null
    grep '^ids: ' "$OUT/gen-$m-$side.err" > "$OUT/gen-$m-$side.ids"
  done
  for f in plan-$m-%s.json plan-$m-%s.txt gen-$m-%s.ids; do
    a=$(printf "$f" main); b=$(printf "$f" br)
    if cmp -s "$OUT/$a" "$OUT/$b"; then echo "$m $a vs br: IDENTICAL ($(wc -c < "$OUT/$a") bytes)"; else echo "$m $a vs br: DIFFER"; diff "$OUT/$a" "$OUT/$b" | head -10; fi
  done
done
