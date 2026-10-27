#!/usr/bin/env bash
# Lane spf: ON spark1. (1) pointer alignment of the first prefill GEMMs (IK_DBG_ALIGN build), (2) rope multi vs off: 64 greedy ids + prefill ms.
export PATH=/usr/local/cuda/bin:$PATH
P=$HOME/intarsiaml-spf/dev; B=$P/target/release/intarsia; O=$P/out/dbg; mkdir -p $O
M=$(ls -d /home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/* | head -1)
head -c 14035 $P/tools/prompts/corpus_docs.txt > $O/p.txt
IK_DBG_ALIGN=1 $B generate -m $M --backend cuda --mem-budget 40 --max-seq 8192 --profile-every 0 --max-tokens 1 -p "$(cat $O/p.txt)" </dev/null 2>&1 | grep -aE "DBG|finish=" | head -30
for mode in 1 0 1 0; do
  INTARSIA_CUDA_ROPE_MULTI=$mode $B generate -m $M --backend cuda --mem-budget 40 --max-seq 8192 --profile-every 0 --temperature 0 --ignore-eos --show-ids --max-tokens 64 -p "$(cat $O/p.txt)" </dev/null > $O/o$mode.out 2> $O/o$mode.err
  echo "ROPE_MULTI=$mode: $(grep -aE 'finish=' $O/o$mode.err | cut -c1-140) ids sha $(grep -a '^ids:' $O/o$mode.err | md5sum | cut -c1-8)"
done
