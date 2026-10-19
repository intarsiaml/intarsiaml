#!/usr/bin/env bash
# Local check of scripts/cuda-round5/mutate.py (no GPU, no build): every mutation applies (its pattern occurs exactly the
# expected number of times) and changes its kernel source.   scripts/cuda-round5/mutate-dryrun.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
bad=0; n=0
for F in kernels/cuda/attention_splitkv.cu kernels/cuda/attention_flash.cu; do
  for m in $(python3 -I "$REPO/scripts/cuda-round5/mutate.py" --list "$(basename "$F")"); do
    rm -rf "$T/kernels"; mkdir -p "$T/kernels/cuda"; cp "$REPO/$F" "$T/$F"
    if ! python3 -I "$REPO/scripts/cuda-round5/mutate.py" "$T" "$m" > /dev/null; then echo "FAIL     $m"; bad=1
    elif cmp -s "$REPO/$F" "$T/$F"; then echo "NOCHANGE $m"; bad=1
    else echo "ok       $m ($(basename "$F"))"; fi
    n=$((n + 1))
  done
done
echo "$n mutations, status $bad"
exit $bad
