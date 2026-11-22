#!/bin/bash
B=./target/release/examples/ink_mix_bench
echo "HEAD $(git rev-parse --short HEAD)"
for cs in 64 128 256; do for rb in 4 16 64; do
  echo "== CS $cs RB $rb"
  INK_BENCH_ONLY=LoopBlend INTARSIA_INK_BLEND_CS=$cs INTARSIA_INK_BLEND_RB=$rb $B --reps 4 --rows 1,16,2048 mix | grep LoopBlend
done; done
echo "== NE4 (row kernel)"
INTARSIA_INK_MIX_NE=4 INK_BENCH_ONLY=ConvMix $B --reps 8 --rows 1,4,16 mix | grep -v blend
INTARSIA_INK_MIX_NE=4 INK_BENCH_ONLY=AttnResMix $B --reps 8 --rows 1,16 mix
echo "== default geometry, ConvMix blend"
INK_BENCH_ONLY=blend $B --reps 8 --rows 1,4,16 mix
