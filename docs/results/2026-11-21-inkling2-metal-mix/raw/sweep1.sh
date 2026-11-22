#!/bin/bash
B=./target/release/examples/ink_mix_bench
for cfg in s8u2 s9u1 s3u3 s1u9; do
  for p in 1 2 4; do
    echo "== cfg $cfg P=$p"
    INTARSIA_INK_PROJ_CFG=$cfg INTARSIA_INK_PROJ_P=$p $B --reps 40 --rows 1 proj | grep InkInProj
  done
  for r in 4 8 16; do
    echo "== cfg $cfg rows $r default"
    INTARSIA_INK_PROJ_CFG=$cfg $B --reps 40 --rows $r proj | grep InkInProj
  done
  echo "== cfg $cfg 8 P2"; INTARSIA_INK_PROJ_CFG=$cfg INTARSIA_INK_PROJ_P=2 $B --reps 40 --rows 8 proj | grep InkInProj
  echo "== cfg $cfg 16 RB8"; INTARSIA_INK_PROJ_CFG=$cfg INTARSIA_INK_PROJ_RB=8 $B --reps 40 --rows 16 proj | grep InkInProj
  echo "== cfg $cfg 16 P2"; INTARSIA_INK_PROJ_CFG=$cfg INTARSIA_INK_PROJ_P=2 $B --reps 40 --rows 16 proj | grep InkInProj
done
$B --reps 40 proj3mm wo
