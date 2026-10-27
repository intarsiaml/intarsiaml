#!/usr/bin/env bash
# Production launcher for spark2's retro role (missioncntrl-guidance), recreated from
# `docker inspect` on 2026-10-22. Only change from the hand-made container: the KV pool is
# sized with --kv-cache-memory-bytes instead of --gpu-memory-utilization 0.50 (which gave a
# 35 GiB pool, 25x the 2 x 131072 tokens --max-num-seqs needs). --gpu-memory-utilization stays: vLLM still
# checks free >= total*util at startup (default 0.92 fails), but no longer sizes the pool. 0.32 = 38.9 GiB,
# just above actual use (~24 weights+non-torch, ~2 activations, 8 KV, 0.6 graphs); keep it close to
# use so a restart does not need memory fleetq has handed to other jobs.
# Image: missioncntrl/vllm-openai:65b7662d = f733c5dd166c (vLLM nightly 65b7662d; its upstream tag was dangling).
set -euo pipefail
KV_BYTES="${KV_BYTES:-8589934592}"
NAME="${NAME:-missioncntrl-guidance}"
docker run -d --name "$NAME" --restart no \
  --gpus all --network host --ipc host --shm-size 16g \
  -v /home/nvidia/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e VLLM_DISABLED_KERNELS=FlashInferFP8ScaledMMLinearKernel \
  --entrypoint vllm \
  missioncntrl/vllm-openai:65b7662d \
  serve nvidia/Qwen3.6-35B-A3B-NVFP4 \
    --host 100.96.95.89 --port 8000 \
    --served-model-name guidance capcom \
    --quantization modelopt_fp4 \
    --max-model-len 131072 --max-num-seqs 16 --max-num-batched-tokens 8192 \
    --kv-cache-dtype fp8 --kv-cache-memory-bytes "$KV_BYTES" \
    --gpu-memory-utilization 0.32 \
    --enable-prefix-caching --enable-prompt-tokens-details --enable-chunked-prefill \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3
