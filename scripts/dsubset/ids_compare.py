#!/usr/bin/env python3
"""Lane mtp-sampled: compare the ids.py results of several server arms.
ids_compare.py <raw dir> <arm>...   (the first arm is the reference)

Per case `name/bN`: every request's ids must equal the reference arm's ids of the same sampling setting at b1 (so the
check covers both the arm and the concurrency). Prints one line per arm and case, the acceptance, and the verdict."""
import json, sys

d, arms = sys.argv[1], sys.argv[2:]
res = {}
for a in arms:
    try:
        res[a] = json.load(open(f"{d}/ids-{a}.json"))
    except OSError:
        print(f"{a}: no result file")
ref = res.get(arms[0])
bad = 0
if ref is None:
    sys.exit("no reference arm")
for a, r in res.items():
    for case, v in r.items():
        name = case.split("/")[0]
        base = ref.get(f"{name}/b1", ref[case])["ids"]
        same = sum(1 for x, y in zip(v["ids"], base) if x == y and x is not None)
        n = len(v["ids"])
        first = [next((k for k, (p, q) in enumerate(zip(x or [], y or [])) if p != q), None) for x, y in zip(v["ids"], base) if x != y]
        m = v["metrics"]
        p, acc, s = m.get("intarsia_spec_proposed_tokens_total", 0), m.get("intarsia_spec_accepted_tokens_total", 0), m.get("intarsia_spec_steps_total", 0)
        print(f"{a:8s} {case:24s} {same}/{n} requests identical to {arms[0]} b1" + (f"  FIRST DIFFS AT {first}" if same != n else "")
              + f" | verify passes {s:.0f} drafted {p:.0f} accepted {acc:.0f}" + (f" ({acc / p:.3f})" if p else ""))
        bad += n - same
print("VERDICT:", "PASS: every request's ids are identical across arms and concurrency" if bad == 0 else f"FAIL: {bad} requests differ")
sys.exit(0 if bad == 0 else 1)
