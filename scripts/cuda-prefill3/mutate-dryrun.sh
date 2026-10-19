#!/usr/bin/env bash
# Apply every mutation of scripts/cuda-prefill3/mutate.py to a scratch copy of kernels/cuda/fp8_gemm.cu (no build): checks
# that each pattern still matches the source exactly as often as declared.  scripts/cuda-prefill3/mutate-dryrun.sh [scratch-dir]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
S=${1:-${TMPDIR:-/tmp}/fp8reg-mutdry}
bad=0
for m in $(python3 "$REPO/scripts/cuda-prefill3/mutate.py" --list); do
  rm -rf "$S"; mkdir -p "$S/kernels/cuda"; cp "$REPO/kernels/cuda/fp8_gemm.cu" "$S/kernels/cuda/"
  python3 "$REPO/scripts/cuda-prefill3/mutate.py" "$S" "$m" || bad=1
  if cmp -s "$REPO/kernels/cuda/fp8_gemm.cu" "$S/kernels/cuda/fp8_gemm.cu"; then echo "$m: no change"; bad=1; fi
done
rm -rf "$S"
exit $bad
