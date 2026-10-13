#!/usr/bin/env python3
"""Reference-free check: within one dump, the prefill row against the decode row of the same position (same tokens, the
decode row comes from a prefill of n - k tokens followed by T = 1 steps). Prints KL21(prefill || decode) per position for
one prompt, and the top-3 of both rows at the given positions.

    uv run --project tools python scripts/acc35b/selfrows.py NAME=DUMP ... --prompt P [--show 55,56]
"""
import argparse, math
import numpy as np
from xcompare import load, kl21

ap = argparse.ArgumentParser()
ap.add_argument("dumps", nargs="+")
ap.add_argument("--prompt", required=True)
ap.add_argument("--show", default="")
a = ap.parse_args()
show = [int(x) for x in a.show.split(",") if x]
for s in a.dumps:
    n, path = s.split("=", 1)
    d = load(path)[0][a.prompt]
    ks = []
    for p in sorted(d["dec"]):
        if p in d["pre"]:
            ks.append((p, kl21(d["pre"][p], d["dec"][p])[0]))
    v = [k for _, k in ks]
    worst = sorted(ks, key=lambda x: -x[1])[:3]
    print(f"{n:8s} KL21(prefill row || decode row): mean {np.mean(v):.4e} max {np.max(v):.4f}  worst " + ", ".join(f"{p}={k:.3f}" for p, k in worst))
    for p in show:
        for lab, (ids, lp) in (("prefill", d["pre"][p]), ("decode", d["dec"][p])):
            o = np.argsort(-lp)[:3]
            print(f"    pos {p} {lab:7s} " + "  ".join(f"{int(ids[j])}:{math.exp(lp[j]):.4f}" for j in o))
