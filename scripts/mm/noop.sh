#!/bin/bash
# Lane mm: Metal batching-off no-op, branch CLI vs origin/main CLI (both --release --features metal), endor, one exclusive slot.
# Usage: scripts/mm/noop.sh BASE_BIN BRANCH_BIN OUTDIR. Per model (Qwen3-8B, 35B): `plan show` text and --json, and 256 greedy
# ids (--show-ids) after a short prompt and a ~4k-token prompt; prints IDENTICAL / DIFFER per pair.
set -u
BASE=$1; BR=$2; OUT=$3; mkdir -p "$OUT"
LONG="$(head -c 16000 /opt/code/wt-mm/tools/prompts/corpus_docs.txt)"
for M in $HOME/models/Qwen3-8B $HOME/models/Qwen3.6-35B-A3B-NVFP4; do
  n=$(basename $M)
  for arm in base branch; do
    bin=$BASE; [ $arm = branch ] && bin=$BR
    $bin plan show -m $M --backend metal --max-seq 8192 > "$OUT/$n-plan-$arm.txt" 2> /dev/null
    $bin plan show -m $M --backend metal --max-seq 8192 --json > "$OUT/$n-plan-$arm.json" 2> /dev/null
    for p in short long; do
      pr="Write a short note about the history of the bicycle."; [ $p = long ] && pr="$LONG"
      $bin generate -m $M --backend metal --max-seq 8192 --temperature 0 --ignore-eos --show-ids --max-tokens 256 -p "$pr" \
        > "$OUT/$n-gen-$p-$arm.out" 2> "$OUT/$n-gen-$p-$arm.err" < /dev/null
      grep -a '^ids:' "$OUT/$n-gen-$p-$arm.err" > "$OUT/$n-ids-$p-$arm.txt"
      grep -a -iE 'prompt tokens|prefill.*tok' "$OUT/$n-gen-$p-$arm.err" | head -1 > "$OUT/$n-ptok-$p-$arm.txt"
    done
  done
  for f in plan-%s.txt plan-%s.json ids-short-%s.txt ids-long-%s.txt gen-short-%s.out gen-long-%s.out; do
    a=$(printf "$OUT/$n-$f" base); b=$(printf "$OUT/$n-$f" branch)
    if cmp -s "$a" "$b" && [ -s "$a" ]; then r=IDENTICAL; else r=DIFFER; fi
    echo "$n $(printf "$f" X): $r ($(wc -c < "$a") / $(wc -c < "$b") B)"
  done
done
