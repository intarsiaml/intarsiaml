#!/usr/bin/env python3
"""Lane rocm102-bisect: compare INTARSIA_DET_TAP files against a reference tap. Usage: tapcmp.py REF.tap OTHER.tap ...
A tap line is `pos=<p> t=<rows> rank=<r> <layer hashes...>` (one token per layer, row hashes joined by commas; det_tap.rs).
Prints, per file and pass, the first layer whose hashes differ from the reference's same pass (or `same`)."""
import sys

def load(p):
    out = []
    for l in open(p):
        f = l.split()
        if not f or not f[0].startswith("pos="):
            continue
        head = [x for x in f if "=" in x and not "," in x and x.split("=")[0] in ("pos", "t", "rank")]
        layers = f[len(head):]
        out.append((" ".join(head), layers))
    return out

ref = load(sys.argv[1])
for p in sys.argv[2:]:
    o = load(p)
    res = []
    for i, (h, ls) in enumerate(o):
        if i >= len(ref):
            res.append(f"[{h}: no ref]"); continue
        rl = ref[i][1]
        d = next((j for j in range(min(len(ls), len(rl))) if ls[j] != rl[j]), None)
        if d is None and len(ls) == len(rl):
            res.append("same")
        else:
            rows = ""
            if d is not None:
                a, b = ls[d].split(","), rl[d].split(",")
                k = next((r for r in range(min(len(a), len(b))) if a[r] != b[r]), None)
                rows = f" row {k}/{len(a)}"
            res.append(f"L{d}{rows}")
    print(f"{p}: {len(o)} passes: {' | '.join(res)}")
