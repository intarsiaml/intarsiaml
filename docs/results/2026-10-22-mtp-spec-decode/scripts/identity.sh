#!/usr/bin/env bash
# Runs ON spark1 inside a fleetq slot: greedy ids --spec off vs --spec mtp (k = 1..3) for a few chat prompts, plus the
# acceptance and decode lines generate prints. Usage: identity.sh <bin> <out dir> [ntok] [ks]
set -uo pipefail
BIN=${1:?bin}; OUT=${2:?out}; N=${3:-256}; KS=${4:-"1 2 3"}
M=$(ls -d $HOME/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
mkdir -p "$OUT"
python3 -c "import json;m=json.load(open('$M/model.safetensors.index.json'))['weight_map'];print('mtp tensors in checkpoint:',len([k for k in m if k.startswith('mtp')]))" | tee "$OUT/checkpoint.txt"
PROMPTS=(
  "Write a Python function that parses an ISO 8601 date string and returns a datetime, with error handling and three unit tests."
  "Explain how a hash map handles collisions, comparing chaining and open addressing, in about 300 words."
  "List the planets of the solar system with one interesting fact about each, as a numbered list."
  "A train leaves at 9:40 and travels 245 km at 70 km/h, then 120 km at 80 km/h. When does it arrive? Show the steps."
)
for i in "${!PROMPTS[@]}"; do
  P=${PROMPTS[$i]}
  for arm in off $(for k in $KS; do echo mtp$k; done); do
    if [[ $arm == off ]]; then SPEC=(--spec off); else SPEC=(--spec mtp --spec-k ${arm#mtp}); fi
    "$BIN" generate --backend cuda --model "$M" --chat --no-think -p "$P" --max-tokens "$N" --ignore-eos --show-ids \
      --max-seq 8192 --mem-budget 30 --profile-every 0 "${SPEC[@]}" > "$OUT/p$i-$arm.out" 2> "$OUT/p$i-$arm.err" < /dev/null
    echo "p$i $arm exit $? $(grep -E 'spec:|decode' "$OUT/p$i-$arm.err" | tr '\n' ' ' | cut -c1-300)"
  done
done
# identity: first differing position of each mtp arm against off
python3 - "$OUT" <<'PY'
import sys, glob, re, os, json
out = sys.argv[1]
def ids(f):
    for l in open(f):
        if l.startswith('ids: '):
            return json.loads(l[5:])
    return None
res = {}
for f in sorted(glob.glob(f'{out}/p*-off.err')):
    p = os.path.basename(f).split('-')[0]
    a = ids(f)
    for g in sorted(glob.glob(f'{out}/{p}-mtp*.err')):
        b = ids(g)
        arm = os.path.basename(g)[len(p)+1:-4]
        if a is None or b is None:
            print(p, arm, 'missing ids'); continue
        d = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        same = d is None and len(a) == len(b)
        print(p, arm, 'IDENTICAL' if same else f'first diff at {d} of {len(a)}')
PY
