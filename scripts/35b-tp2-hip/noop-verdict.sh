#!/usr/bin/env bash
# Lane G2b: no-op verdicts from a window's raw dir (on the Mac): base vs dev accuracy dumps bitwise, accuracy stdout
# identical after dropping the lines that name the build (stamp) or the dump path, plan and greedy ids with cmp.
#   scripts/35b-tp2-hip/noop-verdict.sh docs/results/2026-10-21-35b-tp2-hip/raw/w1
set -uo pipefail
S="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${1:?raw dir}"
FILTER='this build:|wrote self-golden'
for s in 35b-short 35b-long 8b-short 8b-long; do
  echo "dump   $s: $(uv run -q --with numpy --with safetensors python "$S/cmpdump.py" acc-base-$s.safetensors acc-dev-$s.safetensors)"
  if diff <(grep -avE "$FILTER" acc-base-$s.out) <(grep -avE "$FILTER" acc-dev-$s.out) >/dev/null; then
    echo "stdout $s: identical ($(wc -l < acc-dev-$s.out) lines; filtered: $(grep -acE "$FILTER" acc-dev-$s.out) stamp/dump-path lines)"
  else
    echo "stdout $s: DIFFERS"; diff <(grep -avE "$FILTER" acc-base-$s.out) <(grep -avE "$FILTER" acc-dev-$s.out) | head -8
  fi
done
for m in 35b 8b; do
  cmp -s plan-base-$m.out plan-dev-$m.out && echo "plan   $m: identical ($(wc -c < plan-dev-$m.out) bytes)" || echo "plan   $m: DIFFERS"
  for k in short 4k; do
    cmp -s gen-base-$m-$k.ids gen-dev-$m-$k.ids && echo "ids    $m $k: identical (256 ids)" || echo "ids    $m $k: DIFFER"
  done
done
cmp -s gen-base-8b-tp2-short.ids gen-dev-8b-tp2-short.ids && echo "ids    8b tp=2 short: identical (existing HIP tp=2 path)" || echo "ids    8b tp=2 short: DIFFER"
