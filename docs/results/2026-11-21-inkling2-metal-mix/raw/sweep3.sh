#!/bin/bash
B=./target/release/examples/ink_mix_bench
for cfg in s8u2 s9u1 s3u3 s1u9; do
  for p in 1 2; do
    echo "== $cfg P=$p"; INTARSIA_INK_PROJ_CFG=$cfg INTARSIA_INK_PROJ_P=$p $B --reps 10 --rows 1,2,4,8,16 proj | grep "S=3"
  done
  echo "== $cfg 16 RB8"; INTARSIA_INK_PROJ_CFG=$cfg INTARSIA_INK_PROJ_RB=8 $B --reps 10 --rows 16 proj | grep "S=3"
done
$B --reps 10 --rows 1,2,4,8,16 proj3mm wo
