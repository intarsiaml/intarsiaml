#!/usr/bin/env bash
# Round 7: vLLM's own run-to-run KL. Re-score the exact ids of the two 35B vLLM goldens on the live `guidance` server
# (tools/golden_vllm.py --ids-from), from this Mac, and print the live server's version / cache config. vLLM stays up.
#   fleetq run --uses guidance --label "..." -- scripts/cuda-round7/vllm-self.sh <golden dir> <out dir> [extra args]
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
G=${1:?golden dir}; O=${2:?out dir}; shift 2
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
URL=${URL:-http://spark-4e77:8000}
mkdir -p "$O"
echo "== live vLLM version: $(curl -s $URL/version)"
curl -s $URL/metrics | grep -E '^vllm:cache_config_info' | head -1 | cut -c1-400
for s in "" "-long"; do
  uv run --project "$REPO/tools" python "$REPO/tools/golden_vllm.py" --base-url $URL/v1 --model guidance \
    --ids-from "$G/qwen36-35b-a3b-nvfp4-vllm${s}-top20.safetensors" --out "$O/vllm2${s}-top20.safetensors" "$@"
done
