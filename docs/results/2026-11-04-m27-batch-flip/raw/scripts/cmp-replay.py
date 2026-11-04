#!/usr/bin/env python3
"""Compare replay cells: ids, token 184 gap between 14663 and 557, bit equality of top-20 logprob rows against the first cell of each class."""
import json, sys
REF = json.load(open("docs/results/2026-11-03-m27-hs-rowtable/raw/ident/def-mb2-ids.json"))["r1-q4"]["ids"]
A, B = 14663, 557
cells = {}
for f in sys.argv[1:]:
    tag = f.rsplit("/", 1)[-1].split(".")[0]
    for k, v in json.load(open(f)).items():
        cells[f"{tag}/{k}"] = v
def rowmap(row):
    return {int(x[0]): x[1] for x in row}
def gap(v, t=184):
    if not v.get("top"):
        return None
    m = rowmap(v["top"][t])
    return m.get(A), m.get(B)
base = None
print(f"{'cell':32} {'=ref':5} {'1st diff':8} {'id184':6} {'lp(14663)':>11} {'lp(557)':>11} {'gap 14663-557':>14}")
for k, v in cells.items():
    ids = v["ids"]
    d = next((i for i, (a, b) in enumerate(zip(ids, REF)) if a != b), None)
    g = gap(v)
    gs = f"{g[0]:11.6f} {g[1]:11.6f} {g[0] - g[1]:14.6f}" if g and None not in g else "(no logprobs)"
    print(f"{k:32} {str(ids == REF):5} {str(d):8} {ids[184]:<6} {gs}")
print()
lp = [(k, v) for k, v in cells.items() if v.get("top")]
base_k, base_v = lp[0]
print("largest |logprob difference| over shared top-20 ids, rows 0..183, against", base_k)
for k, v in lp[1:]:
    mx, at, bits = 0.0, None, True
    for t in range(184):
        a, b = rowmap(base_v["top"][t]), rowmap(v["top"][t])
        if a != b:
            bits = False
        for i in a.keys() & b.keys():
            dd = abs(a[i] - b[i])
            if dd > mx:
                mx, at = dd, t
    print(f"{k:32} bit-equal rows 0..183: {str(bits):5}  max diff {mx:.6f} at row {at}")
