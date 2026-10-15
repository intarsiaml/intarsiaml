#!/usr/bin/env python3
"""Accuracy table from acc/*.out (and probe/*.out): per KV dtype and golden, prefill / decode top-1 and KL (full-vocabulary for
the short / long goldens, 21-way for the top-20 4k golden), plus the prefix-reuse gate line where present."""
import re, glob, os, sys
R = os.path.dirname(os.path.abspath(__file__))
def rows(f):
    t = open(f).read()
    out = []
    for mode, top1, kind, kl in re.findall(r"(prefill|decode)[^\n]*floor \[bf16\]: top-1 ([\d.]+)%[^\n]*?(full-vocab|21-way) KL ([\d.e-]+)", t):
        out.append((mode, float(top1), float(kl)))
    m = re.search(r"prefix-reuse gate: (\w+)", t)
    return out, (m[1] if m else "-")
if __name__ == "__main__":
    print(f"{'kv':<6} {'golden':<6} {'prefill top-1':>13} {'prefill KL':>11} {'decode top-1':>13} {'decode KL':>10}  reuse")
    order = {"bf16": 0, "fp8": 1, "int8": 2}
    for f in sorted(glob.glob(R + "/acc/*.out"), key=lambda f: (os.path.basename(f).split("-")[1], order.get(os.path.basename(f).split("-")[0], 9))):
        kv, g = os.path.basename(f)[:-4].split("-")
        r, reuse = rows(f)
        d = {m: (a, b) for m, a, b in r}
        p, q = d.get("prefill", (float('nan'),) * 2), d.get("decode", (float('nan'),) * 2)
        print(f"{kv:<6} {g:<6} {p[0]:>13.2f} {p[1]:>11.3e} {q[0]:>13.2f} {q[1]:>10.3e}  {reuse}")
