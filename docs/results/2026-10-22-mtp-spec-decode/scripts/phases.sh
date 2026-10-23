#!/usr/bin/env bash
# Runs ON spark1 inside an exclusive fleetq slot: where a `--spec mtp` pass spends its time. INTARSIA_SPEC_TRACE=1 prints the
# host time of the draft (head passes) and of verify + rollback per pass (both end in a token download); then one
# op-profiled run per arm (`--profile-every 1 --op-table`, eager, synchronising: relative shares only).
# Usage: phases.sh <bin> <out dir>
set -uo pipefail
BIN=${1:?}; OUT=${2:?}; mkdir -p "$OUT"
M=$(ls -d $HOME/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
P="Write a Python function that parses an ISO 8601 date string and returns a datetime, with error handling and three unit tests."
common=(generate --backend cuda --model "$M" --chat --no-think -p "$P" --max-tokens 256 --ignore-eos --max-seq 8192 --mem-budget 30)
for arm in off mtp1 mtp2 mtp3; do
  if [[ $arm == off ]]; then SPEC=(--spec off); else SPEC=(--spec mtp --spec-k ${arm#mtp}); fi
  INTARSIA_SPEC_TRACE=1 "$BIN" "${common[@]}" --profile-every 0 "${SPEC[@]}" > /dev/null 2> "$OUT/trace-$arm.err" < /dev/null
  "$BIN" "${common[@]}" --profile-every 1 --op-table "${SPEC[@]}" > /dev/null 2> "$OUT/ops-$arm.err" < /dev/null
  echo "$arm: $(grep -oE 'decode [0-9]+ ms \([0-9.]+ tok/s\)' "$OUT/trace-$arm.err") $(grep -oE 'spec: verify.*' "$OUT/trace-$arm.err")"
  python3 - "$OUT/trace-$arm.err" <<'PY'
import re, sys
d, v = [], []
for l in open(sys.argv[1]):
    m = re.search(r'mtp draft ([0-9.]+) ms, verify \+ rollback ([0-9.]+) ms', l)
    if m:
        d.append(float(m.group(1))); v.append(float(m.group(2)))
if d:
    d.sort(); v.sort()
    print(f"  passes {len(d)}: draft median {d[len(d)//2]:.2f} ms mean {sum(d)/len(d):.2f}; verify+rollback median {v[len(v)//2]:.2f} ms mean {sum(v)/len(v):.2f}")
PY
done
