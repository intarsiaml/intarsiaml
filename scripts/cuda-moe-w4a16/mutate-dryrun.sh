#!/usr/bin/env bash
# Local dry run of mutate.py: every mutation must apply to the current kernel source exactly as declared.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
T="$(mktemp -d)"
mkdir -p "$T/kernels/cuda"
bad=0
for m in $(python3 "$REPO/scripts/cuda-moe-w4a16/mutate.py" --list); do
  cp "$REPO/kernels/cuda/moe_w4a16.cu" "$T/kernels/cuda/"
  python3 "$REPO/scripts/cuda-moe-w4a16/mutate.py" "$T" "$m" | cut -c1-120 || bad=1
done
rm -rf "$T"
exit $bad
