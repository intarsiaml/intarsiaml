#!/usr/bin/env python3
"""Lane TP2R: the tables of a window run, from the `intarsia bench speed` JSONs in
<window dir>/<arm>-r<round>/tp2r-<arm>-r<round>.json.
  - A/B table: per arm and size, the median over rounds of each round's median prefill and decode tok/s, per-round values,
    run-to-run spread (min..max over rounds), and deltas of the tp=2 arms against tp1g.
  - streams table: aggregate decode tok/s per stream count, per arm (rows with `concurrency` in the JSON).
    ab_table.py <window dir>"""
import glob
import json
import os
import statistics as st
import sys

d = sys.argv[1]
cells, conc = {}, {}
for f in sorted(glob.glob(f"{d}/*-r*/tp2r-*.json")):
    arm = os.path.basename(os.path.dirname(f)).rsplit("-r", 1)[0]
    j = json.load(open(f))
    for r in j["rows"]:
        if r.get("reps_ok", 0) == 0:
            continue
        c = cells.setdefault((arm, r["target_prompt_tokens"]), {"dec": [], "pre": [], "ttft": []})
        c["dec"].append(r["decode_tok_s"])
        c["pre"].append(r["prefill_tok_s"])
        c["ttft"].append(r["ttft_s"])
    for r in j.get("concurrency") or []:
        conc.setdefault((arm, r["streams"]), []).append(r)
arms = sorted({a for a, _ in cells if a not in ("tp1s", "tp2s", "tp2mtp")})  # the streams / mtp arms run other configs
sizes = sorted({s for a, s in cells if a in arms})
if cells:
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
    for key, name, fmt in (("pre", "prefill", "{:.0f}"), ("dec", "decode", "{:.1f}")):
        print(f"{a} {name} per round: " + "; ".join(
            f"{s}: " + ", ".join(fmt.format(v) for v in cells[(a, s)][key]) + f" (spread {100 * (max(cells[(a, s)][key]) / min(cells[(a, s)][key]) - 1):.1f} %)"
            for s in sizes if (a, s) in cells))
print()
for x, y in [("tp2g", "tp1g"), ("tp2gf", "tp1g"), ("tp2gf", "tp2g"), ("tp1m", "tp1g"), ("tp2g", "tp1m"), ("tp2gf", "tp1m")]:
    if x in med and y in med:
        pd = [100 * (p / q - 1) for p, q in zip(med[x][0], med[y][0])]
        dd = [100 * (p / q - 1) for p, q in zip(med[x][1], med[y][1])]
        print(f"{x} vs {y}: prefill " + " / ".join(f"{v:+.1f} %" for v in pd) + "; decode " + " / ".join(f"{v:+.1f} %" for v in dd))
if conc:
    print()
    print("| arm | streams | per-stream tok/s | aggregate tok/s | TTFT s |")
    print("|---|---|---|---|---|")
    for (a, n), rows in sorted(conc.items()):
        def m(k):
            v = [r[k] for r in rows if r.get(k) is not None]
            return st.median(v) if v else float("nan")
        print(f"| {a} | {n} | {m('per_stream_decode_tok_s'):.1f} | {m('aggregate_decode_tok_s'):.1f} | {m('ttft_s'):.2f} |")
