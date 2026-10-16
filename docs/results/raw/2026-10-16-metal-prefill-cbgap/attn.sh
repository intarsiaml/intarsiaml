#!/bin/zsh
# Job B: 256-row prefill attention against a long KV (Qwen3-8B shape 32/8 heads, head_dim 128, 36 layers). No model load.
SC=/private/tmp/claude-501/-opt-code-intarsiaml/0e8026c5-6dd0-4b83-b375-5ff0a66894fc/scratchpad/cbgap
R=/opt/code/intarsiaml-cbgap/docs/results/raw/2026-10-16-metal-prefill-cbgap
O=$R/attn; mkdir -p $O
source $R/waitidle.sh
wait_idle || touch $O/contaminated
macmon pipe -i 200 > $O/macmon.jsonl 2>/dev/null & MP=$!
$SC/pso_nax_attn 256 2048 240 > $O/pso.txt 2>&1
# per-chunk-size cost at fixed history
for P in 3840 1024; do
  for T in 128 192 224 240 256 272 288 320 384 512 1024 2048; do
    echo "== pos0 $P chunk $T"; $SC/flash_bench --chunk $T --reps 8 nax attn_nax_d128_s4 attn_nax_d128_s8 $P
  done
done > $O/chunk-sweep.txt 2>&1
# whole 4062-token prompt: sum over its chunks
for T in 256 512 1024 2048; do
  $SC/flash_bench --chunk $T --reps 8 --total 4062 attn_nax_d128_s4 attn_nax_d128_s8
done > $O/total-4062.txt 2>&1
# per position
for T in 256 2048; do
  $SC/flash_bench --chunk $T --reps 8 attn_nax_d128_s4 0 512 1024 2048 3072 3840
done > $O/pos-sweep.txt 2>&1
kill $MP 2>/dev/null
echo done
