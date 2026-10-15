#!/bin/zsh
set -eu
S=/private/tmp/claude-501/-opt-code-intarsiaml/31e10c55-edcb-4c6c-8a85-99bab193d028/scratchpad/fwd1
M=/Users/josef/Models/Qwen3-8B
G=/opt/code/intarsiaml/golden
cd $S
# 1-row tail probe: long golden prompt of 300 tokens at chunk 299 -> (299,1) before, (298,2) after; reference = single chunk
for b in before after; do
  ./intarsia-$b accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --max-seq 8192 --prefill-chunk 299 > probe-$b-299.out 2> probe-$b-299.err
done
./intarsia-after accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --max-seq 8192 --prefill-chunk 2048 > probe-after-2048.out 2> probe-after-2048.err
# the three goldens at default chunk, after binary (compare to recorded qualify numbers)
./intarsia-after accuracy --model $M --backend metal --golden $G/qwen3-8b-fp32.safetensors --mode both --decode-positions 16 --max-seq 4096 > acc-short.out 2> acc-short.err
./intarsia-after accuracy --model $M --backend metal --golden $G/qwen3-8b-long-fp32.safetensors --mode both --decode-positions 64 --max-seq 8192 > acc-long.out 2> acc-long.err
./intarsia-after accuracy --model $M --backend metal --golden $G/qwen3-8b-4k-fp32-top20.safetensors --mode both --decode-positions 64 --max-seq 8192 --min-top1 0 > acc-4k.out 2> acc-4k.err
echo done
