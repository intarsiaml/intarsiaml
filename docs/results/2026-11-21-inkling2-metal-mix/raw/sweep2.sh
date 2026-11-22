#!/bin/bash
B=./target/release/examples/ink_mix_bench
for cfg in s8u2 s9u1 s3u3; do
 for gc in 80 160 320 640 1280 100000; do
  echo "== $cfg grid $gc"
  INTARSIA_INK_PROJ_CFG=$cfg INTARSIA_INK_PROJ_GRID=$gc $B --reps 30 --rows 1,4 proj | grep "S=1"
 done
done
