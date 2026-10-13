#!/usr/bin/env bash
set -u
D=$HOME/intarsia-dbg; BIN=$D/src/target/dbg/release/intarsia
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
SIMT=attn_decode_bf16_splitkv,attn_prefill_bf16_splitkv,attn_prefill_bf16_flash
FLAGS="--allow-unstamped --allow-dirty --allow-same-code"
cd $D/out
echo "##### 1. attention sweep (with the added 4631-token chat shapes)"; date +%T
(cd $D/src/crates/intarsia-backend-cuda && INTARSIA_REQUIRE_GPU=1 $D/src/target/dbg/release/deps/attention-1f3f7fe05bbcd025 attention_kernels_match_reference_on_random_shapes --nocapture 2>&1 | tail -4 | cut -c1-300)
echo "##### 1b. mutated flash mask (key > lim)"; date +%T
(cd $D/mut/crates/intarsia-backend-cuda && INTARSIA_REQUIRE_GPU=1 $D/mut/target/mut/release/deps/attention-1f3f7fe05bbcd025 attention_kernels_match_reference_on_random_shapes --nocapture 2>&1 | tail -6 | cut -c1-400)
echo "##### 2. 35B: the same document with and without the ChatML wrapper"; date +%T
G=$D/g/wrap.safetensors
COMMON="--model $M35 --backend cuda --max-seq 8192 --mem-budget 70 --mode prefill $FLAGS"
rm -f pos_w_*.tsv
INTARSIA_ACC_POS_DUMP=pos_w_flash256.tsv $BIN accuracy $COMMON --prefill-chunk 256 --golden $G --write-self-golden w_flash256.safetensors > log_w_flash256.txt 2>&1
grep -E "^(A_|B_|C_)|OVERALL" log_w_flash256.txt | cut -c1-200
INTARSIA_CUDA_DISABLE=$SIMT INTARSIA_ACC_POS_DUMP=pos_w_fs256.tsv $BIN accuracy $COMMON --prefill-chunk 256 --self-golden w_flash256.safetensors --self-max-kl 1e9 --self-max-pos-kl 1e9 --self-max-flip-pct 100 --self-max-tail 1e9 > log_w_fs256.txt 2>&1
echo "-- SIMT vs flash (same chunk)"; grep -E "^(A_|B_|C_)|OVERALL" log_w_fs256.txt | cut -c1-200
INTARSIA_ACC_POS_DUMP=pos_w_flash1024.tsv $BIN accuracy $COMMON --prefill-chunk 1024 --golden $G --write-self-golden w_flash1024.safetensors > log_w_flash1024.txt 2>&1
echo "-- flash chunk 1024"; grep -E "^(A_|B_|C_)|OVERALL" log_w_flash1024.txt | cut -c1-200
echo "##### 3. fp32 HF goldens for Qwen3-8B"; date +%T
mkdir -p $D/gold; ls $D/gold
MC=/hf/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
docker run --rm --name dbg-golden-8b --gpus all --entrypoint python3 --user $(id -u):$(id -g) -e HOME=/tmp -e HF_HUB_OFFLINE=1 -v /home/nvidia/.cache/huggingface:/hf:ro -v $D/gold:/work vllm/vllm-openai:v0.28.0-aarch64-cu129 /work/golden.py --model $MC --out /work/qwen3-8b-fp32.safetensors --device cuda --dtype float32 --prompts /work/prompts/golden.jsonl 2>&1 | tail -12
date +%T
docker run --rm --name dbg-golden-8b-long --gpus all --entrypoint python3 --user $(id -u):$(id -g) -e HOME=/tmp -e HF_HUB_OFFLINE=1 -v /home/nvidia/.cache/huggingface:/hf:ro -v $D/gold:/work vllm/vllm-openai:v0.28.0-aarch64-cu129 /work/golden.py --model $MC --out /work/qwen3-8b-long-fp32.safetensors --device cuda --dtype float32 2>&1 | tail -14
date +%T; ls -la $D/gold/*.safetensors
echo "##### 4. 8B CUDA accuracy"; date +%T
C8="--model $M8 --backend cuda --max-seq 4096 --mem-budget 40 --mode both --decode-positions 16 $FLAGS"
for g in $D/gold/qwen3-8b-fp32.safetensors $D/gold/qwen3-8b-long-fp32.safetensors $HOME/intarsiaml-golden/qwen3-8b.safetensors $HOME/intarsiaml-golden/qwen3-8b-long.safetensors; do
  b=$(basename $g .safetensors); echo "=== $b"
  $BIN accuracy $C8 --golden $g > log_8b_$b.txt 2>&1
  grep -E "^golden:|== mode|^[a-z_0-9]+ +[0-9]+ +[0-9]+ +[0-9.]+ |OVERALL|floor \[|floor verdict" log_8b_$b.txt | cut -c1-230
done
date +%T
