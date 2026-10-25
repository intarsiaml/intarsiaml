#!/usr/bin/env bash
# Lane l5 no-op gate (knob OFF): origin/main build against the branch build: `plan show` (text, json) and 256 greedy ids after a short and a ~3.5k-token
# prompt, 35B and 27B, each pair on the same card one after the other. The knob is not set (and the 27B has no routed experts at all).
. "$(dirname "$0")/lib.sh" 2>/dev/null || . "$HOME/intarsiaml-l5-$NAME/scripts/l5/lib.sh"
PD=$HOME/intarsiaml-l5-$NAME/scripts/hg/prompts
echo "[noop] base rev $(cat $HOME/intarsiaml-l5-base/.rev); branch snapshot $NAME $(now)"
for m in 35 27; do
  if [ $m = 35 ]; then M=$M35; else M=$M27; fi
  for who in base dev; do
    fn=bs; [ $who = dev ] && fn=dv
    for p in short 4k; do gen $fn n$m-$p-$who "$M" "$PD/$p.txt" 256; done
    $fn /target/release/intarsia plan show --backend hip --model "$M" --max-seq 8192 --json > "$O/plan-$m-$who.json" 2> "$O/plan-$m-$who.json.err"
    $fn /target/release/intarsia plan show --backend hip --model "$M" --max-seq 8192 > "$O/plan-$m-$who.txt" 2> "$O/plan-$m-$who.txt.err"
  done
done
echo "== no-op comparison (base = origin/main $(cat $HOME/intarsiaml-l5-base/.rev | cut -c1-9))"
for m in 35 27; do
  for p in short 4k; do
    if cmp -s "$O/gen-n$m-$p-base.ids" "$O/gen-n$m-$p-dev.ids" && [ -s "$O/gen-n$m-$p-dev.ids" ]; then echo "ids $m $p: identical ($(tr ',' '\n' < "$O/gen-n$m-$p-dev.ids" | wc -l) ids, md5 $(md5sum < "$O/gen-n$m-$p-dev.ids" | cut -c1-8))"
    else echo "ids $m $p: DIFFER or empty"; fi
  done
  for k in json txt; do
    if cmp -s "$O/plan-$m-base.$k" "$O/plan-$m-dev.$k" && [ -s "$O/plan-$m-dev.$k" ]; then echo "plan $m $k: byte-identical ($(wc -c < "$O/plan-$m-dev.$k") B)"
    else echo "plan $m $k: DIFFER or empty"; fi
  done
done
echo "[noop] done $(now)"
