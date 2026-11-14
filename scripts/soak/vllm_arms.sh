#!/usr/bin/env bash
# soak lane, vLLM arms while production vLLM is up. Run under:
#   FLEETQ_OWNER=claude-soak fleetq run --host spark2 --quiet gpu,cpu --expect N --label "soak: vLLM arms" -- bash scripts/soak/vllm_arms.sh <step...>
# steps: k3k4 (replay2 K=3 seed 1, K=4 seed 2; records the traces the intarsia arms replay), think <limit> <max-tokens> (GSM8K thinking on)
# Every arm is bracketed by vLLM's own request counters: any request beyond ours is production traffic inside the measurement.
set -uo pipefail
WLOG=vllm-arms
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
[[ -f $OUT/replay-plan.json ]] || cp "$REPO/docs/results/2026-11-13-interleave/raw/replay-plan.json" "$OUT/replay-plan.json"
bracket() { # bracket <name> <own-requests> <cmd...>
  local name=$1 own=$2; shift 2
  local b a; b=$(vcount); log "$name: vLLM counters before: $b (own expected: $own)"
  "$@"
  a=$(vcount); log "$name: vLLM counters after:  $a"
  echo "$name before: $b | after: $a | own requests (excl. idle probes): $own" >> "$OUT/vllm-foreign-traffic.txt"
}
while [[ $# -gt 0 ]]; do
  case $1 in
    k3k4)
      shift
      bracket V-k3 49 replay2 V-k3 "$VURL" interleaved 1 3 "$OUT/replay-trace-seed1.json"
      bracket V-k4 49 replay2 V-k4 "$VURL" interleaved 2 4 "$OUT/replay-trace-seed2.json" ;;
    pilot) # pilot <n> <max-tokens>: sizes the thinking-on arm (tokens per problem, time per problem); not part of the results
      RC=$(recipe_file V-pilot "vLLM production (see V-think)")
      bracket V-pilot "$2" gsm8k V-pilot "$VURL" "$RC" on "$2" "$3"; shift 3 ;;
    think)
      RC=$(recipe_file V-think "docker missioncntrl/vllm-openai:65b7662d (vllm 0.26.1rc1.dev602): vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 --served-model-name guidance capcom --quantization modelopt_fp4 --max-model-len 131072 --max-num-seqs 2 --max-num-batched-tokens 8192 --kv-cache-dtype fp8 --enable-prefix-caching --enable-chunked-prefill --tool-call-parser qwen3_xml --reasoning-parser qwen3 (no speculative config)")
      bracket V-think "$2" gsm8k V-think "$VURL" "$RC" on "$2" "$3"; shift 3 ;;
    *) echo "unknown step $1"; exit 2 ;;
  esac
done
