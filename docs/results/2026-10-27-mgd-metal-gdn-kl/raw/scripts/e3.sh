#!/bin/bash
cd /opt/code/intarsiaml/.worktrees/mgd
M=$HOME/models/Qwen3.6-35B-A3B-NVFP4; G=/opt/code/intarsiaml/golden
export INTARSIA_METAL_FP8_NAX=0 INTARSIA_METAL_MOE_NAX=0
for dp in 16 100000; do
export INTARSIA_ACC_POS_DUMP=$PWD/run/e3/pos$dp.tsv; rm -f $INTARSIA_ACC_POS_DUMP
./target/release/intarsia accuracy --model $M --backend metal --golden $G/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors --mode both --decode-positions $dp --max-seq 4096 --allow-unstamped --write-self-golden run/e3/self$dp.safetensors > run/e3/log$dp.txt 2>&1
echo exit $?
done
