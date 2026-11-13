#!/usr/bin/env bash
# ab35b arm A: production vLLM, measured live while it is up. Run under: FLEETQ_OWNER=claude-ab35b fleetq run --host spark2 --quiet gpu,cpu ...
set -uo pipefail
: "${FLEETQ_JOB:?run under fleetq}"
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
URL=http://$HN:8000/v1
RC=$(recipe_file A "docker missioncntrl/vllm-openai:65b7662d (vllm 0.26.1rc1.dev602): vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --served-model-name guidance capcom --quantization modelopt_fp4 --max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --enable-prefix-caching --enable-chunked-prefill --tool-call-parser qwen3_xml --reasoning-parser qwen3 (no speculative config)")
echo "A start $(date +%T)  vLLM /v1/models: $(curl -s -o /dev/null -w '%{http_code}' $URL/models)" | tee -a "$OUT/arm_a.log"
replay A "$URL" interleaved 0
replay A "$URL" sequential 1
gsm8k A1 "$URL" "$RC"
gsm8k A2 "$URL" "$RC"
sampled A "$URL"
tools A "$URL"
curl -s "http://$HN:8000/metrics" | grep -E 'prefix_cache|^vllm:cache_config' | head -8 > "$OUT/A-vllm-prefixcache.metrics"
echo "A done $(date +%T)" | tee -a "$OUT/arm_a.log"
