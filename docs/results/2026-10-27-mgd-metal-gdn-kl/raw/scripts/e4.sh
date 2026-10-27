#!/bin/bash
cd /opt/code/intarsiaml/.worktrees/mgd
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; G=/opt/code/intarsiaml/golden
./target/release/intarsia accuracy --model $M --backend metal --golden $G/oracle/qwen36-35b-a3b-nvfp4-oracle-set-vllm-top20.safetensors --mode both --decode-positions 100000 --max-seq 8192 --prefill-chunk 256 --allow-unstamped --write-self-golden run/e4/self.safetensors > run/e4/log.txt 2>&1
echo exit $?
