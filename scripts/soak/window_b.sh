#!/usr/bin/env bash
# soak lane, window B (spark2, vLLM down): GSM8K quality arms on the production-shaped serve. usage: window_b.sh <cap-min> <think-n> <think-max-tokens>
# Runs INSIDE: FLEETQ_OWNER=claude-soak fleetq run --host spark2 --takes-down retro --quiet gpu,cpu --mem 58 --expect 60 ...
#   (a) Ba: --max-batch 2 --prefix-store on, bf16 KV, GSM8K n=200 thinking off max_tokens 1024 pinned (ab35b Run 2 setup)
#   (b) Bb: same + --kv-dtype fp8
#   (c) Bc: same as (a), GSM8K thinking on, n=<think-n>, max_tokens=<think-max-tokens>, the same values as the vLLM arm (V-think)
# Arm (c) takes the n that still fits in the remaining time (at most <think-n>, at PER_PROBLEM_S seconds per problem, default 30; skipped below 30).
set -uo pipefail
CAPMIN=${1:-50}; TN=${2:-100}; TMAX=${3:-8192}
WLOG=window-b
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
take_vllm_down "$CAPMIN"
[[ $(rsh 'systemctl is-active earlyoom') == active ]] || log "WARNING: earlyoom NOT active"
left() { echo $(( CAPMIN*60 - ($(date +%s)-T0) )); }
RA=$(recipe_file Ba "intarsia serve --backend cuda (HF snapshot nvidia/Qwen3.6-35B-A3B-NVFP4 1355db6a) --max-seq 131072 --mem-budget 40 --max-batch 2 --prefix-store on --profile-every 0; bf16 KV, no MTP (the store refuses drafting), prefill chunk 2048, tool parser hermes")
RB=$(recipe_file Bb "intarsia serve --backend cuda (HF snapshot nvidia/Qwen3.6-35B-A3B-NVFP4 1355db6a) --max-seq 131072 --mem-budget 40 --max-batch 2 --prefix-store on --kv-dtype fp8 --profile-every 0; fp8_e4m3 KV, no MTP, prefill chunk 2048, tool parser hermes")
RC=$(recipe_file Bc "intarsia serve --backend cuda (HF snapshot nvidia/Qwen3.6-35B-A3B-NVFP4 1355db6a) --max-seq 131072 --mem-budget 40 --max-batch 2 --prefix-store on --profile-every 0; bf16 KV, no MTP, thinking on")

if start_serve Ba --max-batch 2 --prefix-store on; then
  gsm8k Ba "$URL" "$RA" off 200 1024
  finish_serve Ba
fi
if start_serve Bb --max-batch 2 --prefix-store on --kv-dtype fp8; then
  gsm8k Bb "$URL" "$RB" off 200 1024
  finish_serve Bb
fi
# Bc: n = what still fits at PER_PROBLEM_S per problem (vLLM thinking arm measured ~29 s: ~2200 tokens at 77 tok/s; the 5-problem pilot said 19.4 s) after 4 min of margin for start and restore
N=$(( ($(left) - 240) / ${PER_PROBLEM_S:-30} )); (( N > TN )) && N=$TN
if (( N >= 30 )); then
  log "Bc: n=$N (asked $TN), $(left) s left"
  if start_serve Bc --max-batch 2 --prefix-store on; then
    gsm8k Bc "$URL" "$RC" on "$N" "$TMAX"
    finish_serve Bc
  fi
else
  log "skipping Bc: $(left) s left (n would be $N)"
fi
