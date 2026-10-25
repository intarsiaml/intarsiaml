#!/usr/bin/env bash
# Lane hg: batching-off no-op check of one binary, or the comparison of two.
#   RUN=<command prefix that runs the intarsia binary> noop.sh run <outdir> <tag> <backend> <model8> <model35> <promptdir> [extra flags]
#   noop.sh cmp <outdir> <tagA> <tagB>
# `run` writes, per model (8 = Qwen3-8B, 35 = Qwen3.6-35B-A3B-NVFP4): `plan show` (text and --json) and the 256 greedy ids
# after a short prompt (400 chars) and after a ~3.5k-token prompt. `cmp` prints one line per artifact: identical or DIFFER.
set -uo pipefail
mode=$1; shift
if [ "$mode" = run ]; then
  OUT=$1; TAG=$2; BE=$3; M8=$4; M35=$5; PD=$6; shift 6
  mkdir -p "$OUT"
  for m in 8 35; do
    if [ "$m" = 8 ]; then M=$M8; else M=$M35; fi
    for p in short 4k; do
      $RUN generate -m "$M" --backend "$BE" --max-seq 8192 "$@" --chat --no-think --temperature 0 --profile-every 0 --ignore-eos \
        --show-ids -p "$(cat "$PD/$p.txt")" --max-tokens 256 < /dev/null > /dev/null 2> "$OUT/gen-$m-$p-$TAG.err"
      grep -a '^ids:' "$OUT/gen-$m-$p-$TAG.err" > "$OUT/gen-$m-$p-$TAG.ids"
      echo "[noop] $TAG $m $p: $(tr ',' '\n' < "$OUT/gen-$m-$p-$TAG.ids" | wc -l) ids"
    done
    $RUN plan show --backend "$BE" --model "$M" --max-seq 8192 "$@" --json < /dev/null > "$OUT/plan-$m-$TAG.json" 2> "$OUT/plan-$m-$TAG.json.err"
    $RUN plan show --backend "$BE" --model "$M" --max-seq 8192 "$@" < /dev/null > "$OUT/plan-$m-$TAG.txt" 2> "$OUT/plan-$m-$TAG.txt.err"
  done
else
  OUT=$1; A=$2; B=$3
  for m in 8 35; do
    for p in short 4k; do
      f="$OUT/gen-$m-$p"
      if cmp -s "$f-$A.ids" "$f-$B.ids" && [ -s "$f-$A.ids" ]; then
        echo "ids $m $p: identical ($(tr ',' '\n' < "$f-$B.ids" | wc -l | tr -d ' ') ids, sha $(shasum "$f-$B.ids" | cut -c1-8))"
      else echo "ids $m $p: DIFFER or empty"; fi
    done
    for k in json txt; do
      f="$OUT/plan-$m"
      if cmp -s "$f-$A.$k" "$f-$B.$k" && [ -s "$f-$A.$k" ]; then echo "plan $m $k: byte-identical ($(wc -c < "$f-$B.$k" | tr -d ' ') B)"
      else echo "plan $m $k: DIFFER or empty"; fi
    done
  done
fi
