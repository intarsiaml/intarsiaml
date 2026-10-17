#!/usr/bin/env bash
# Local sanity check of scripts/cuda-decode/mutate.py: every mutation must apply (its pattern occurs exactly once) to a
# throwaway copy of kernels/. No GPU, no build:   scripts/cuda-decode/mutcheck.sh [mutation ...]
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ALL=$(grep -o 'ALL="[^"]*"' "$REPO/scripts/cuda-decode/mutate.sh" | sed 's/ALL="//; s/"$//')
MUTS=${*:-$ALL}
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
bad=0
for m in $MUTS; do
  rm -rf "$T/kernels"; cp -R "$REPO/kernels" "$T/kernels"
  if python3 -I "$REPO/scripts/cuda-decode/mutate.py" "$T" "$m" >/dev/null; then
    if diff -rq "$REPO/kernels" "$T/kernels" >/dev/null; then echo "NOCHANGE $m"; bad=1; else echo "ok       $m"; fi
  else echo "FAIL     $m"; bad=1; fi
done
exit $bad
