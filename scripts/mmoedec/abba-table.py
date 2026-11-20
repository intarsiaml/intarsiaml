#!/usr/bin/env python3
"""mmoedec: tables of the A B B A rounds (raw/ab/{offA,onB,onC,offD,onA,offB,offC,onD}-r<odd>.jsonl): per round and cell, the median decode tok/s of the
six reps of each arm (two servers x 3 reps), the on/off ratio per round and its median over rounds; and by position (A..D) the median over rounds."""
import json, glob, os, re, statistics as st, sys
d = sys.argv[1]
cells = ["chat-short", "code-gen", "reason", "agent-edit"]
data = {}  # (round, arm group, pos, cell) -> [tps]
for f in glob.glob(os.path.join(d, "*-r*.jsonl")):
    m = re.match(r"(off|on)([ABCD])-r(\d+)\.jsonl$", os.path.basename(f))
    if not m: continue
    g, pos, r = m.group(1), m.group(2), int(m.group(3))
    for l in open(f):
        try: j = json.loads(l)
        except Exception: continue
        if "dec_tps" in j and j.get("cell") in cells: data.setdefault((r, g, pos, j["cell"]), []).append(j["dec_tps"])
rounds = sorted({k[0] for k in data})
print("| cell | off | on | on/off (median of per-round ratios; per round) |\n|---|---|---|---|")
for c in cells:
    offs, ons, ratios = [], [], []
    for r in rounds:
        o = [v for k, vs in data.items() if k[0] == r and k[1] == "off" and k[3] == c for v in vs]
        n = [v for k, vs in data.items() if k[0] == r and k[1] == "on" and k[3] == c for v in vs]
        if o and n: offs += o; ons += n; ratios.append(st.median(n) / st.median(o))
    print(f"| {c} | {st.median(offs):.1f} [{min(offs):.0f}-{max(offs):.0f}] | {st.median(ons):.1f} [{min(ons):.0f}-{max(ons):.0f}] | **{st.median(ratios):.3f}x** ({', '.join('%.3f' % x for x in ratios)}) |")
print("\nBy server position in the round (median tok/s over rounds and reps; position A is the first server of a round):\n")
print("| cell | A | B | C | D |\n|---|---|---|---|---|")
for c in cells:
    row = []
    for p in "ABCD":
        v = [x for k, vs in data.items() if k[2] == p and k[3] == c for x in vs]
        row.append(f"{st.median(v):.1f}" if v else "-")
    print(f"| {c} | " + " | ".join(row) + " |")
