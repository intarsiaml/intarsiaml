#!/usr/bin/env bash
# On the Spark (run via ssh inside a fleetq slot). start-sglang.sh <model: 35b|8b> <port>
# SGLang image lmsysorg/sglang:qwen38-27b-dflash2 (sglang 0.0.0.dev0+qwen38.27b.g561c8f3, torch 2.13+cu130). Container fx3-sglang.
set -u
M=${1:?35b|8b}; PORT=${2:-8200}
IMG=lmsysorg/sglang:qwen38-27b-dflash2
docker rm -f fx3-sglang >/dev/null 2>&1
snap() { local d; d=$(ls -d $HOME/.cache/huggingface/hub/models--$1/snapshots/* | head -1); echo "/root/.cache/huggingface/${d#*/.cache/huggingface/}"; }
case $M in
  35b) MODEL=$(snap nvidia--Qwen3.6-35B-A3B-NVFP4); EXTRA="--quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 --mem-fraction-static ${MEMFRAC:-0.36}";;
  8b)  MODEL=$(snap Qwen--Qwen3-8B); EXTRA="--mem-fraction-static ${MEMFRAC:-0.33}";;
esac
docker run -d --name fx3-sglang --gpus all --ipc=host --network host \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface -e HF_HUB_OFFLINE=1 --entrypoint python3 $IMG \
  -m sglang.launch_server --model-path $MODEL --host 0.0.0.0 --port $PORT --served-model-name sglang \
  --context-length 20480 --max-running-requests ${MAXREQ:-16} --trust-remote-code $EXTRA ${SGL_ARGS:-} >/dev/null
echo started
