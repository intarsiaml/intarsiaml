#!/usr/bin/env bash
# remote: position-level diagnosis of chat_reuse_4k. Runs on spark1 inside the exclusive lease (vLLM down).
set -u
D=$HOME/intarsia-dbg; BIN=$D/src/target/dbg/release/intarsia
M=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
G=$D/g/dbg.safetensors
SIMT=attn_decode_bf16_splitkv,attn_prefill_bf16_splitkv,attn_prefill_bf16_flash
COMMON="--model $M --backend cuda --max-seq 8192 --mem-budget 70 --mode prefill --allow-unstamped --allow-dirty --allow-same-code"
mkdir -p $D/out; cd $D/out
for CH in ${CHUNKS:-256}; do
  rm -f pos_flash_$CH.tsv pos_simt_$CH.tsv pos_fs_$CH.tsv
  echo "== flash chunk $CH vs vLLM"; date +%T
  INTARSIA_ACC_POS_DUMP=pos_flash_$CH.tsv $BIN accuracy $COMMON --prefill-chunk $CH --golden $G --write-self-golden self_flash_$CH.safetensors > log_flash_$CH.txt 2>&1
  tail -18 log_flash_$CH.txt | cut -c1-200
  if [ -z "${NOSIMT:-}" ]; then
  echo "== simt chunk $CH vs vLLM"; date +%T
  INTARSIA_CUDA_DISABLE=$SIMT INTARSIA_ACC_POS_DUMP=pos_simt_$CH.tsv $BIN accuracy $COMMON --prefill-chunk $CH --golden $G > log_simt_$CH.txt 2>&1
  tail -18 log_simt_$CH.txt | cut -c1-200
  echo "== simt chunk $CH vs flash self-golden"; date +%T
  INTARSIA_CUDA_DISABLE=$SIMT INTARSIA_ACC_POS_DUMP=pos_fs_$CH.tsv $BIN accuracy $COMMON --prefill-chunk $CH --self-golden self_flash_$CH.safetensors --self-max-kl 1e9 --self-max-pos-kl 1e9 --self-max-flip-pct 100 --self-max-tail 1e9 > log_fs_$CH.txt 2>&1
  tail -18 log_fs_$CH.txt | cut -c1-200
  fi
done
date +%T
