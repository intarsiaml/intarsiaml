#!/bin/bash
cd /opt/code/intarsiaml/.worktrees/ink-m-mix
B=./target/release/examples/ink_mix_bench
echo "HEAD $(git rev-parse --short HEAD)"
F="plain,blend S=3,LoopBlend"
for round in 1 2; do
  echo "== round $round old (NE 3, gate 16 rows)"
  INTARSIA_INK_MIX_NE=3 INTARSIA_INK_BLEND_RB=16 INK_BENCH_ONLY="$F" $B --reps 5 --rows 1,16 mix | grep -E "R=(1|16|2048) "
  echo "== round $round new (NE 4, gate 4 rows)"
  INK_BENCH_ONLY="$F" $B --reps 5 --rows 1,16 mix | grep -E "R=(1|16|2048) "
done
echo "== end"
