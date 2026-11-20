#!/usr/bin/env python3
"""samphost: ABBA gate table from the window JSONs (sampled.py `decode_each`), pooled per arm, with 2 SE of the difference of means.
  python3 scripts/samphost/gate.py RAW_DIR T "none=a1,a4" "fast=a2,a3"
Also prints the per-window means and the MTP tokens per verify pass."""
import json, math, pathlib, statistics as st, sys

raw = pathlib.Path(sys.argv[1]); T = sys.argv[2]
arms = [(a.split("=")[0], a.split("=")[1].split(",")) for a in sys.argv[3:]]

def rows(w):
    f = raw / f"{w}-T{T}-sampled.json"
    return {r["size"]: r for r in json.load(open(f))["rows"]} if f.exists() else {}

sizes = sorted({s for _, ws in arms for w in ws for s in rows(w)})
print(f"T={T}: per window, mean decode tok/s (reps)")
print("| window | " + " | ".join(str(s) for s in sizes) + " |\n|---|" + "---|" * len(sizes))
pool = {}
for name, ws in arms:
    for w in ws:
        r = rows(w)
        print(f"| {name} {w} | " + " | ".join(f"{st.mean(r[s]['decode_each']):.1f} ({len(r[s]['decode_each'])})" if s in r else "-" for s in sizes) + " |")
        for s in sizes:
            if s in r:
                pool.setdefault((name, s), []).extend(r[s]["decode_each"])
for name, _ in arms:
    print(f"| **{name} pooled** | " + " | ".join(f"{st.mean(pool[(name, s)]):.1f} ({len(pool[(name, s)])})" if (name, s) in pool else "-" for s in sizes) + " |")
if len(arms) == 2:
    a, b = arms[0][0], arms[1][0]
    cells = []
    for s in sizes:
        x, y = pool.get((a, s)), pool.get((b, s))
        if not x or not y or len(x) < 2 or len(y) < 2:
            cells.append("-"); continue
        d = st.mean(y) / st.mean(x) - 1
        se = math.sqrt(st.variance(x) / len(x) + st.variance(y) / len(y)) / st.mean(x)
        cells.append(f"**{d * 100:+.1f} %** (2 SE {2 * se * 100:.1f} %)")
    print(f"| {b} vs {a} | " + " | ".join(cells) + " |")
