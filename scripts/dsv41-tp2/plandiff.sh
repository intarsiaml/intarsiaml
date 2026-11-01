#!/usr/bin/env bash
# Lane dsv41/tp2: `intarsia plan show` of the 35B NVFP4 and Qwen3-8B on CUDA, base (BASE_REF, default dsv41/plan) vs this
# branch's binary (~/intarsia-dsv41-tp2/bin/intarsia, from build.sh). Builds the base on spark2 in ~/intarsia-dsv41-tp2/src-base.
# EXL3=1 also diffs main's EXL3 checkpoints (Qwen3-8B EXL3 4.0 bpw, plain and bf16plain, on spark1): plan show and 32 greedy
# tokens with each binary (run inside a --quiet gpu,cpu slot on spark1 as well: the job must hold spark1 and spark2).
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
if [[ ${EXL3:-0} == 1 ]]; then
  r1() { ssh -n -o BatchMode=yes spark1 "export FLEETQ_JOB=$FLEETQ_JOB; $*"; }
  rsh 'scp -q ~/intarsia-dsv41-tp2/src-base/target/release/intarsia spark1:intarsia-dsv41-tp2/bin/intarsia-base'
  for d in qwen3-8b-exl3-4.0bpw qwen3-8b-exl3-4.0bpw-bf16plain; do
    m=/home/nvidia/models-m2/$d
    for b in base tp2; do
      bin='$HOME/intarsia-dsv41-tp2/bin/intarsia'; [[ $b == base ]] && bin='$HOME/intarsia-dsv41-tp2/bin/intarsia-base'
      r1 "$bin plan show -m $m --backend cuda 2>&1" > "$OUT/$d-$b.txt" || true
      r1 "$bin generate -m $m --backend cuda -p 'The sky appears blue because' --max-tokens 32 2>&1 | grep -vE 'loaded|finish=|steps=|prefill chunk'" > "$OUT/$d-$b-gen.txt" || true
    done
    if diff -q <(grep -v -iE 'built|stamp|took|loaded in|ms\b' "$OUT/$d-base.txt") <(grep -v -iE 'built|stamp|took|loaded in|ms\b' "$OUT/$d-tp2.txt") >/dev/null; then
      echo "$d: plan show unchanged ($(wc -l < "$OUT/$d-tp2.txt") lines)"
    else
      echo "$d: plan show DIFFERS"; { diff "$OUT/$d-base.txt" "$OUT/$d-tp2.txt" | head -20; true; }
    fi
    if cmp -s "$OUT/$d-base-gen.txt" "$OUT/$d-tp2-gen.txt"; then echo "$d: greedy 32 tokens identical: $(head -c 120 "$OUT/$d-tp2-gen.txt" | tr '\n' ' ')"; else echo "$d: greedy DIFFERS"; diff "$OUT/$d-base-gen.txt" "$OUT/$d-tp2-gen.txt" | head; fi
  done
fi
