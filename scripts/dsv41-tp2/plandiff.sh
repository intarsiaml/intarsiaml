#!/usr/bin/env bash
# Lane dsv41/tp2: `intarsia plan show` of the 35B NVFP4 and Qwen3-8B on CUDA, base (BASE_REF, default dsv41/plan) vs this
# branch's binary (~/intarsia-dsv41-tp2/bin/intarsia, from build.sh). Builds the base on spark2 in ~/intarsia-dsv41-tp2/src-base.
#   fleetq run --host spark2 --load cpu --mem 20 --expect 25 --label "dsv41 tp2 plan diff" -- scripts/dsv41-tp2/plandiff.sh
set -euo pipefail
: "${FLEETQ_JOB:?run under fleetq run}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BASE=${BASE_REF:-dsv41/plan}
OUT=$REPO/.scratch/plandiff; mkdir -p "$OUT"
RP='export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH; export MAX_JOBS=4 CARGO_BUILD_JOBS=4;'
rsh() { ssh -n -o BatchMode=yes spark2 "$RP export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
rsh 'mkdir -p ~/intarsia-dsv41-tp2/src-base; test -d ~/intarsia-dsv41-tp2/src-base/target || cp -a ~/intarsia-dsv41-tp2/src/target ~/intarsia-dsv41-tp2/src-base/target'
git -C "$REPO" archive --format=tar "$BASE" | ssh -o BatchMode=yes spark2 'tar -C ~/intarsia-dsv41-tp2/src-base -xf -'
echo "base $(git -C "$REPO" rev-parse --short "$BASE")"
rsh 'cd ~/intarsia-dsv41-tp2/src-base && cargo build --release --features cuda -p intarsia-cli 2>&1 | tail -2'
M35=/home/nvidia/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/1355db6a052410cfd62085d94b58866fd0f2c3c5
M8=/home/nvidia/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218
for m in 35b:$M35 8b:$M8; do
  n=${m%%:*}; d=${m#*:}
  rsh "~/intarsia-dsv41-tp2/src-base/target/release/intarsia plan show -m $d --backend cuda 2>&1" > "$OUT/$n-base.txt" || true
  rsh "~/intarsia-dsv41-tp2/bin/intarsia plan show -m $d --backend cuda 2>&1" > "$OUT/$n-tp2.txt" || true
  if diff -q <(grep -v -iE 'built|stamp|took|loaded in|ms\b' "$OUT/$n-base.txt") <(grep -v -iE 'built|stamp|took|loaded in|ms\b' "$OUT/$n-tp2.txt") >/dev/null; then
    echo "$n: plan show unchanged ($(wc -l < "$OUT/$n-tp2.txt") lines)"
  else
    echo "$n: plan show DIFFERS"; { diff "$OUT/$n-base.txt" "$OUT/$n-tp2.txt" | head -40; true; }
  fi
done
