#!/usr/bin/env bash
. "$(dirname "$0")/common.sh"
B=/home/josef/bench-r9700/llama.cpp-v0.5.0-hip/build/bin
docker run -d --name fx6-llamacpp-hip --network host $DEV --user "$(id -u):$(id -g)" -e HOME=/tmp -e HIP_VISIBLE_DEVICES=0 \
  -e LD_LIBRARY_PATH=/lcpp:/opt/rocm/lib -v $B:/lcpp:ro -v $GG:/gguf:ro --entrypoint /lcpp/llama-server fx6-rocm-llama:7.2.4 \
  -m /gguf/Qwen3.6-35B-A3B-MXFP4_MOE.gguf --alias MXFP4_MOE -ngl 999 -fa on -c 20480 --cache-ram 0 -np 1 --host 0.0.0.0 --port 8091
