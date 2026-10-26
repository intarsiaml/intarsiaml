#!/usr/bin/env python3
"""Lane lb: summarise the localeval outputs of ab.sh (~/.cache/lb-ab/<tag>-<n>-<tree>.out) into base vs dev tables.
usage: abtab.py <dir> <tag>"""
import glob
import re
import statistics as st
import sys

d, tag = sys.argv[1], sys.argv[2]
arms = {"base": [], "dev": []}
for f in sorted(glob.glob(f"{d}/{tag}-*.out")):
    m = re.search(r"-(\d+)-(base|dev)\.out$", f)
    if not m:
        continue
    txt = open(f).read()
    rows = {}
    for r in re.finditer(r"\[localeval\]\s+(\d+) tok rep \d+/\d+: prompt=(\d+).*?prefill=(\d+) tok/s decode=([\d.]+) tok/s", txt):
        rows.setdefault(int(r.group(1)), []).append((float(r.group(3)), float(r.group(4))))
    agg = re.search(r"8 streams: aggregate ([\d.]+) tok/s", txt)
    arms[m.group(2)].append({s: (st.median(p for p, _ in v), st.median(q for _, q in v)) for s, v in rows.items()} | {"agg8": float(agg.group(1)) if agg else None})
sizes = [512, 4096, 16384]
print("| metric | base (3 runs, median of medians; runs) | dev | delta |")
print("|---|---|---|---|")
def line(name, key, idx):
    vals = {}
    for t in ("base", "dev"):
        vs = [a[key][idx] if idx is not None else a[key] for a in arms[t]]
        vals[t] = (st.median(vs), vs)
    b, dv = vals["base"], vals["dev"]
    fmt = lambda x: ", ".join(f"{v:.1f}" for v in x)
    print(f"| {name} | {b[0]:.1f} ({fmt(b[1])}) | {dv[0]:.1f} ({fmt(dv[1])}) | {(dv[0]/b[0]-1)*100:+.2f} % |")
for s in sizes:
    line(f"prefill tok/s @ {s}", s, 0)
for s in sizes:
    line(f"decode tok/s @ {s} ctx, 1 stream", s, 1)
line("decode aggregate tok/s, 8 streams (4096-token prompts)", "agg8", None)
