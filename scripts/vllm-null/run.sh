#!/usr/bin/env bash
# vllm-null lane: rerun the soak's vLLM thinking-on GSM8K arm (V-think) to measure vLLM-vs-vLLM spread.
#   fleetq run --host spark2 --load gpu --expect 90 --label "vllm-null: vLLM thinking-on rerun" -- bash scripts/vllm-null/run.sh <n> V2 [V3 ...]
# Same call as scripts/soak/vllm_arms.sh `think`: localeval quality --thinking on --structured prompt --pin-gsm8k --max-tokens 8192 (sequential, temperature 0), but --limit <n> (85: the ids Bc ran).
set -uo pipefail
WLOG=vllm-null
export SOAK_OUT=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/docs/results/2026-11-13-vllm-null/raw
source "$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)/scripts/soak/lib.sh"
N=$1; shift
for arm in "$@"; do
  RC=$(recipe_file "$arm" "docker missioncntrl/vllm-openai:65b7662d (vllm 0.26.1rc1.dev602): vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --served-model-name guidance capcom --quantization modelopt_fp4 --max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --enable-prefix-caching --enable-chunked-prefill --tool-call-parser qwen3_xml --reasoning-parser qwen3 (no speculative config); rerun of V-think")
  b=$(vcount); log "$arm: vLLM counters before: $b (own expected: $N)"
  gsm8k "$arm" "$VURL" "$RC" on "$N" 8192
  a=$(vcount); log "$arm: vLLM counters after:  $a"
  echo "$arm before: $b | after: $a | own requests: $N" >> "$OUT/vllm-foreign-traffic.txt"
done
