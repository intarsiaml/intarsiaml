#!/usr/bin/env python3
"""Lane G1c: the A/B table of a window run: per arm and size, the median over rounds of each round's median decode and
prefill (`intarsia bench speed --at-most` JSONs in <window dir>/<arm>-r<round>/g1c-<arm>-r<round>.json).
    ab_table.py <window dir>"""
import glob
import json
import os
import statistics as st
import sys

d = sys.argv[1]
cells = {}
for f in sorted(glob.glob(f"{d}/*-r*/g1c-*.json")):
    arm = os.path.basename(os.path.dirname(f)).rsplit("-r", 1)[0]
    j = json.load(open(f))
    for r in j["rows"]:
        if r.get("reps_ok", 0) == 0:
            continue
        c = cells.setdefault((arm, r["target_prompt_tokens"]), {"dec": [], "pre": [], "ttft": [], "graph": []})
        c["dec"].append(r["decode_tok_s"])
        c["pre"].append(r["prefill_tok_s"])
        c["ttft"].append(r["ttft_s"])
        g = r.get("graph") or {}
        c["graph"].append(g.get("replays"))
arms = sorted({a for a, _ in cells})
sizes = sorted({s for _, s in cells})
print("| arm | rounds | " + " | ".join(f"prefill {s}" for s in sizes) + " | " + " | ".join(f"decode {s}" for s in sizes) + " |")
print("|---|---|" + "---|" * (2 * len(sizes)))
med = {}
for a in arms:
    n = max(len(cells.get((a, s), {"dec": []})["dec"]) for s in sizes)
    pre = [st.median(cells[(a, s)]["pre"]) if (a, s) in cells else float("nan") for s in sizes]
    dec = [st.median(cells[(a, s)]["dec"]) if (a, s) in cells else float("nan") for s in sizes]
    med[a] = (pre, dec)
    print(f"| {a} | {n} | " + " | ".join(f"{v:.0f}" for v in pre) + " | " + " | ".join(f"{v:.1f}" for v in dec) + " |")
print()
for a in arms:
    print(f"{a} decode per round: " + "; ".join(f"{s}: {cells[(a, s)]['dec']}" for s in sizes if (a, s) in cells))
print()
for x, y in [("tp2g", "tp1g"), ("tp2g", "tp2e"), ("tp2e", "tp1g")]:
    if x in med and y in med:
        pd = [100 * (p / q - 1) for p, q in zip(med[x][0], med[y][0])]
        dd = [100 * (p / q - 1) for p, q in zip(med[x][1], med[y][1])]
        print(f"{x} vs {y}: prefill " + " / ".join(f"{v:+.1f} %" for v in pd) + "; decode " + " / ".join(f"{v:+.1f} %" for v in dd))
