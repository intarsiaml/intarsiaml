#!/usr/bin/env bash
# Offline KL21 table of round 7 (Mac, no fleet): every candidate truncated to top-20, all rows and with system/user-turn
# rows excluded (scripts/m3-accuracy/setkl.py, the M3 method). Pairs: stored vLLM golden vs vLLM's own re-run, dev default,
# lt (opt-in), lt at other chunks; and dev vs lt (the change's own size).
#   scripts/cuda-round7/kl-table.sh <golden dir> <vllm re-run dir> <dump dir>
set -euo pipefail
G=${1:?golden dir}; V=${2:?vllm re-run dir}; D=${3:?dump dir}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
K="uv run --project $REPO/tools python $REPO/scripts/m3-accuracy/setkl.py"
echo "== 35b-short, reference = stored vLLM golden"
$K "$G/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors" "$V/vllm2-top20.safetensors" "$D/self-dev-short-c2048.safetensors" \
  "$D/self-lt-short-c2048.safetensors" "$D/self-lt-short-c256.safetensors"
echo "== 35b-short, reference = dev default (chunk 2048)"
$K "$D/self-dev-short-c2048.safetensors" "$D/self-lt-short-c2048.safetensors" "$D/self-lt-short-c256.safetensors"
echo "== 35b-long, reference = stored vLLM golden"
$K "$G/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors" "$V/vllm2-long-top20.safetensors" "$D/self-dev-long-c2048.safetensors" \
  "$D/self-lt-long-c2048.safetensors" "$D/self-lt-long-c256.safetensors" "$D/self-lt4k-long-c4096.safetensors"
echo "== 35b-long, reference = dev default (chunk 2048)"
$K "$D/self-dev-long-c2048.safetensors" "$D/self-lt-long-c2048.safetensors" "$D/self-lt-long-c256.safetensors" "$D/self-lt4k-long-c4096.safetensors"
