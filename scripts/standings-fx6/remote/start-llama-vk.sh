#!/usr/bin/env bash
# usage: start-llama-vk.sh v0.5.0|v0.4.0   (Vulkan/RADV llama-server on MXFP4_MOE, port 8091)
V=${1:-v0.5.0}
if [ "$V" = v0.5.0 ]; then B=/home/josef/bench-r9700/llama.cpp-v0.5.0-vulkan/build/bin; else B=/home/josef/.local/opt/llama.cpp; fi
M=/data/models/gguf/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-MXFP4_MOE.gguf
GGML_VK_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=$B nohup $B/llama-server -m $M --alias MXFP4_MOE -ngl 999 -fa on -c 20480 --cache-ram 0 -np 1 --host 0.0.0.0 --port 8091 >/tmp/fx6-llama.log 2>&1 </dev/null &
echo $! >/tmp/fx6-llama.pid
