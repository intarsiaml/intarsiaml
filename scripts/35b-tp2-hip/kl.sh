#!/usr/bin/env bash
# Lane G2b: KL21 table (round-7 / G1b method, scripts/m3-accuracy/setkl.py, candidates truncated to top-20) on the Mac.
#   scripts/35b-tp2-hip/kl.sh <gold dir: vLLM short/long goldens + m3 long-A/long-B re-runs> <raw dir>
# rows: vLLM re-run vs its golden (long-A/B; reference noise), tp=1 vs vLLM, tp=2 vs vLLM, tp=2 vs tp=1 (size of the change)
set -uo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
G=${1:?gold dir}; R=${2:?raw dir}
K() { uv run -q --with numpy --with safetensors python "$S/m3-accuracy/setkl.py" "$@" 2>&1 | cut -c1-150; }
echo "== short: ref vLLM golden; cands tp=1 (w1 dev), tp=2 (w1)"
K "$G/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors" "$R/w1/acc-dev-35b-short.safetensors" "$R/w1/acc-dev-35b-tp2-short.safetensors"
echo "== short: ref tp=1; cand tp=2"
K "$R/w1/acc-dev-35b-short.safetensors" "$R/w1/acc-dev-35b-tp2-short.safetensors"
echo "== long: ref vLLM golden; cands vLLM re-runs long-A, long-B, tp=1 (w1 dev), tp=2 (w2)"
K "$G/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors" "$G/long-A.safetensors" "$G/long-B.safetensors" "$R/w1/acc-dev-35b-long.safetensors" "$R/w2/tp2-long.safetensors"
echo "== long: ref tp=1; cand tp=2"
K "$R/w1/acc-dev-35b-long.safetensors" "$R/w2/tp2-long.safetensors"
