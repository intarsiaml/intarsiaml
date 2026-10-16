#!/bin/bash
# usage: acc.sh <extra args...>   (cpu, Qwen3-Reranker-0.6B, long golden)
M=~/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473
cd /opt/code/intarsiaml
exec ./target/release/intarsia accuracy --model $M --backend cpu --golden golden/qwen3-reranker-0.6b-long.safetensors --mode both --decode-positions 16 --allow-unstamped "$@"
