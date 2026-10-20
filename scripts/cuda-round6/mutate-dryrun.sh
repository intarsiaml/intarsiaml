#!/usr/bin/env bash
# Apply every scripts/cuda-round6/mutate.py mutation to a scratch copy of the sources (no build): each must apply exactly.
#   scripts/cuda-round6/mutate-dryrun.sh [scratch dir]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
D=${1:-$(mktemp -d)}
rm -rf "$D"; mkdir -p "$D/kernels/cuda" "$D/crates/intarsia-backend-cuda/src"
bad=0
for m in $(python3 -I "$REPO/scripts/cuda-round6/mutate.py" --list); do
  cp "$REPO/kernels/cuda/attention_flash.cu" "$D/kernels/cuda/"
  cp "$REPO/crates/intarsia-backend-cuda/src/quantmoe.rs" "$D/crates/intarsia-backend-cuda/src/"
  python3 -I "$REPO/scripts/cuda-round6/mutate.py" "$D" "$m" | cut -c1-110 || bad=1
done
rm -rf "$D"
exit $bad
