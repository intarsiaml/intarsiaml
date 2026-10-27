#!/bin/bash
# G1 extra: more decode positions (64 per prompt) on short and long, for base and candidate, one slot.
OUT=$1; shift; mkdir -p $OUT
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; G=/opt/code/intarsiaml/golden
for a in "$@"; do tag=${a%%=*}; bin=${a#*=}
  $bin accuracy --model $M --backend metal --golden $G/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions 64 --max-seq 4096 --write-verdict $OUT/$tag-short.verdict.json > $OUT/$tag-short.log 2>&1; echo "$tag short exit $?"
  $bin accuracy --model $M --backend metal --golden $G/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors --mode both --decode-positions 64 --max-seq 16384 --write-verdict $OUT/$tag-long.verdict.json > $OUT/$tag-long.log 2>&1; echo "$tag long exit $?"
done
