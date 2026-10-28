#!/usr/bin/env python3
"""Lane vrc: per-category, per-round pooled decode tok/s of a cat-window's arms (same pooling as cat_table.py), the new/old ratio per
round, and the plain-decode speed rows. usage: python3 -I tools/dfh/vrc_spread.py <raw window dir> [new old]... (pairs as arm names)"""
import json, os, re, sys, collections
raw = sys.argv[1].rstrip("/"); res = "/opt/code/deepeval/results"
pairs = [(sys.argv[i], sys.argv[i + 1]) for i in range(2, len(sys.argv) - 1, 2)] or [("mtp2", "mtp2-old"), ("dfl7", "dfl7-old")]
steps = open(os.path.join(raw, "steps.log")).read()
runs = re.findall(r"round (\d+) arm (\S+) (\S+): localeval exit (\d+) (results/\S+)", steps)
def pooled(cases):
    toks = sum(x["speed"]["completion_tokens"] - 1 for x in cases)
    secs = sum((x["speed"]["completion_tokens"] - 1) / x["speed"]["decode_tok_s"] for x in cases if x["speed"]["decode_tok_s"])
    return toks / secs
d = collections.defaultdict(dict)
for r, arm, cat, rc, path in runs:
    cases = json.load(open(os.path.join(os.path.dirname(res), path)))["datasets"][0]["cases"]
    d[(arm, cat)][int(r)] = pooled(cases)
cats = sorted({c for _, c in d})
print("| category | arm | round 1 | round 2 | round 3 | median | spread (max/min-1) |\n|---|---|---:|---:|---:|---:|---:|")
for c in cats:
    for arm in sorted({a for a, _ in d}, key=lambda a: (a != "off", a)):
        v = d.get((arm, c))
        if not v: continue
        xs = [v[k] for k in sorted(v)]
        med = sorted(xs)[len(xs) // 2]
        print(f"| {c} | {arm} | " + " | ".join(f"{x:.1f}" for x in xs) + " | " * (3 - len(xs)) + f" {med:.1f} | {100 * (max(xs) / min(xs) - 1):.1f}% |")
print("\n| category | pair | ratio per round | median |\n|---|---|---|---:|")
for c in cats:
    for new, old in pairs:
        a, b = d.get((new, c)), d.get((old, c))
        if not a or not b: continue
        rs = [a[k] / b[k] for k in sorted(a) if k in b]
        print(f"| {c} | {new} / {old} | " + ", ".join(f"{x:.3f}" for x in rs) + f" | {sorted(rs)[len(rs) // 2]:.3f} |")
