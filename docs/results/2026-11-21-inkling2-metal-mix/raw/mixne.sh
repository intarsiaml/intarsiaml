#!/bin/bash
B=./target/release/examples/ink_mix_bench
$B psoinfo
for ne in 3 4 9; do echo "== NE $ne"; INTARSIA_INK_MIX_NE=$ne $B --reps 8 --rows 1 mix 2>&1 | grep -E "R=1 |R=2048" | grep -E "plain|AttnResMix N=4|SconvRes"; done
