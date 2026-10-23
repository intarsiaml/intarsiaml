#!/usr/bin/env bash
# Runs ON a Spark inside a `--quiet gpu,cpu` fleetq slot: INTARSIA_MTP_VERIFY_CHECK=1 row-mismatch measurement (Fable
# review finding 1). Every --spec mtp verify pass's rows are also run as single-row decode steps from the same state;
# per row: max |dlogit|, whether the argmax agrees, the single-row top-2 logit margin, and whether the row decided an
# emitted token (every earlier draft accepted).
# Prompts: the 4 identity prompts (256 tokens) and two high-entropy prose prompts (2048 tokens), k = 2.
# Usage: verify-check.sh <bin> <out dir>
set -uo pipefail
BIN=${1:?}; OUT=${2:?}; mkdir -p "$OUT"
M=$(ls -d $HOME/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4/snapshots/* | head -1)
P=(
  "Write a Python function that parses an ISO 8601 date string and returns a datetime, with error handling and three unit tests."
  "Explain how a hash map handles collisions, comparing chaining and open addressing, in about 300 words."
  "List the planets of the solar system with one interesting fact about each, as a numbered list."
  "A train leaves at 9:40 and travels 245 km at 70 km/h, then 120 km at 80 km/h. When does it arrive? Show the steps."
  "Write a long, original short story about a lighthouse keeper who finds a stranger's diary washed ashore. Use vivid sensory detail, dialogue, and an unexpected ending."
  "Write a reflective personal essay on what cities sound like at night, wandering freely between memories, history, music and philosophy."
)
N=(256 256 256 256 2048 2048)
for i in "${!P[@]}"; do
  INTARSIA_MTP_VERIFY_CHECK=1 "$BIN" generate --backend cuda --model "$M" --chat --no-think -p "${P[$i]}" --max-tokens "${N[$i]}" \
    --ignore-eos --max-seq 8192 --mem-budget 30 --profile-every 0 --spec mtp --spec-k 2 > "$OUT/vc-p$i.out" 2> "$OUT/vc-p$i.err" < /dev/null
  echo "p$i exit $? rows $(grep -c '^spec-check' "$OUT/vc-p$i.err") $(grep -oE 'spec: verify.*' "$OUT/vc-p$i.err")"
done
python3 - "$OUT" <<'PY'
import sys, glob, json
out = sys.argv[1]
allrows = []
per = {}
for f in sorted(glob.glob(f'{out}/vc-p*.err')):
    p = f.split('/')[-1][3:-4]
    for l in open(f):
        if l.startswith('spec-check: rows'):
            for r, cell in enumerate(l.split()[3:]):
                d, ok, m, dec = cell.split(':')
                row = (p, r, float(d), ok == '1', float(m), dec == '1')
                allrows.append(row); per.setdefault(p, []).append(row)
def summ(rows):
    ds = sorted(x[2] for x in rows); n = len(rows)
    q = lambda f: ds[min(n - 1, int(f * n))]
    mism = [x for x in rows if not x[3]]
    return {"rows": n, "bitwise_rows": sum(d == 0 for d in ds), "dlogit_p50": q(.5), "dlogit_p90": q(.9), "dlogit_p99": q(.99),
            "dlogit_max": ds[-1], "argmax_mismatch": len(mism), "mismatch_rate": len(mism) / n,
            "margin_lt_1e-2": sum(x[4] < 1e-2 for x in rows), "mismatch_rows": [(x[0], x[1], x[2], x[4]) for x in mism][:20]}
dec = [x for x in allrows if x[5]]
res = {"all": summ(allrows), "deciding": summ(dec), **{p: summ(v) for p, v in per.items()},
       "by_row": {str(r): summ([x for x in allrows if x[1] == r]) for r in sorted({x[1] for x in allrows})}}
json.dump(res, open(f'{out}/summary.json', 'w'), indent=1)
a = res["all"]
print(f"ALL rows {a['rows']} bitwise {a['bitwise_rows']} max|dlogit| p50 {a['dlogit_p50']:.3e} p90 {a['dlogit_p90']:.3e} p99 {a['dlogit_p99']:.3e} max {a['dlogit_max']:.3e} argmax mismatch {a['argmax_mismatch']} ({100*a['mismatch_rate']:.3f}%) margin<1e-2 {a['margin_lt_1e-2']}")
d = res["deciding"]
print(f"DECIDING rows (all earlier drafts accepted) {d['rows']} argmax mismatch {d['argmax_mismatch']} ({100*d['mismatch_rate']:.3f}%) margin<1e-2 {d['margin_lt_1e-2']} max|dlogit| {d['dlogit_max']:.3e}")
for p, v in per.items():
    print(f"  p{p}: rows {v['rows']} mismatch {v['argmax_mismatch']} margin<1e-2 {v['margin_lt_1e-2']} max {v['dlogit_max']:.3e}")
PY
