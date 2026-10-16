set -u
B=/opt/code/intarsiaml/target/release/intarsia
G=/opt/code/intarsiaml/golden/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors
for spec in "mut105:attn_prefill,attn_decode:scale=1.05" "mut101:attn_prefill,attn_decode:scale=1.01" "moe101:moe_experts:scale=1.01"; do
  tag=${spec%%:*}; m=${spec#*:}
  echo "== $tag $m $(date +%T)"
  INTARSIA_CPU_MUTATE=$m INTARSIA_CPU_FREE_MEM_GB=20 $B accuracy --model ~/models/Qwen3.6-35B-A3B-NVFP4 --backend cpu --golden $G --mode both --decode-positions 16 --max-seq 4096 --mem-budget 60 --prefill-chunk 256 --allow-unstamped --allow-dirty --write-baseline cpu-$tag.baseline.json --write-self-golden cpu-$tag.dump.safetensors > cpu-$tag.log 2>&1
  echo "exit $? $(date +%T)"; grep -E "^OVERALL" cpu-$tag.log
done
