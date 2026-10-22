#!/usr/bin/env bash
# Lane H1: window-1 offline analysis on the Mac: bitwise dump comparisons and the KL21 table (scripts/m3-accuracy/setkl.py).
#   scripts/h1-tp2-hip/analyze-w1.sh <dumps dir> <golden dir>  > docs/results/2026-10-21-h1-tp2-hip/raw/w1/kl-and-dumps.txt
set -uo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${1:?dumps}"; GD=${2:?goldens}
C() { uv run -q --with numpy --with safetensors python "$S/h1-tp2-hip/cmpdump.py" "$@"; }
K() { uv run -q --with numpy --with safetensors python "$S/m3-accuracy/setkl.py" "$@" 2>&1 | cut -c1-200; }
for p in "base-35b-short dev-35b-short" "base-8b-short dev-8b-short" "base-35b-tp2-short dev-35b-tp2g-short" \
  "base-35b-tp2-long dev-35b-tp2g-long" "dev-35b-tp2g-short dev-35b-tp2gf-short" "base-8b-tp2-short dev-8b-tp2g-short" \
  "base-35b-tp2-short diag-tp2-commhost-short" "base-35b-short trace-tp1-short" "base-35b-tp2-short trace-tp2-short" \
  "base-35b-short cal-gemv48-short" "base-35b-short cal-chunk128-short" "base-35b-short cal-simt-short"; do
  set -- $p; echo "dump $1 vs $2: $(C acc-$1.safetensors acc-$2.safetensors)"
done
M=$GD/qwen36-35b-a3b-nvfp4-vllm-top20-meta.safetensors
echo "== short: ref vLLM golden (-meta, cd6ef94b); cands: other vLLM run (b58845c3), tp1, tp2, tp1 simt, tp2 simt, tp1 chunk128, tp2 bf16 partials"
K $M $GD/qwen36-35b-a3b-nvfp4-vllm-top20.safetensors acc-base-35b-short.safetensors acc-base-35b-tp2-short.safetensors \
  acc-cal-simt-short.safetensors acc-diag-tp2-simt-short.safetensors acc-cal-chunk128-short.safetensors acc-diag-tp2-partbf16-short.safetensors
echo "== short: ref tp1 base; cands: tp2, tp2 bf16 partials, tp1 simt, tp1 chunk128, tp1 gemv48"
K acc-base-35b-short.safetensors acc-base-35b-tp2-short.safetensors acc-diag-tp2-partbf16-short.safetensors \
  acc-cal-simt-short.safetensors acc-cal-chunk128-short.safetensors acc-cal-gemv48-short.safetensors
echo "== short: ref tp1 simt; cand tp2 simt (both without split-KV / flash attention)"
K acc-cal-simt-short.safetensors acc-diag-tp2-simt-short.safetensors
echo "== long: ref vLLM golden; cands tp1, tp2"
K $GD/qwen36-35b-a3b-nvfp4-vllm-long-top20.safetensors acc-base-35b-long.safetensors acc-base-35b-tp2-long.safetensors
echo "== long: ref tp1; cand tp2"
K acc-base-35b-long.safetensors acc-base-35b-tp2-long.safetensors
