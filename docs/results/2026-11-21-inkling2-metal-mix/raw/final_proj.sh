#!/bin/bash
B=./target/release/examples/ink_mix_bench
echo "HEAD $(git rev-parse --short HEAD) bin $(stat -f %Sm $B)"
scripts/metal-speed-gate/refbw.py
$B --reps 12 --rows 1,2,4,8,16 proj proj3mm wo
$B --reps 12 --rows 2,4,8,12,16 naxsmall
$B --reps 8 prefill
scripts/metal-speed-gate/refbw.py
