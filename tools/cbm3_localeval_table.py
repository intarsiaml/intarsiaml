#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Gate (b) table from a `cbm3_slot.sh localeval` directory: copies each localeval result.json next to the logs
(DIR/results/<arm>-<round>.json) and prints, per concurrency level, aggregate_decode_tok_s (tokens / wall) and overlap_decode_tok_s per
arm and round with the paired ratio num/den (default on/off).   cbm3_localeval_table.py DIR [--num on --den off] [--md OUT.md]"""
import argparse
import json
import re
import shutil
import statistics as st
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("dir")
ap.add_argument("--num", default="on")
ap.add_argument("--den", default="off")
ap.add_argument("--md")
a = ap.parse_args()
d = Path(a.dir)
(d / "results").mkdir(exist_ok=True)
data = {}  # arm -> round -> {streams: (agg, overlap)}
for f in sorted(d.glob("localeval-*-*.log")):
    m = re.match(r"localeval-(.+)-(\d+)\.log", f.name)
    arm, rd = m.group(1), int(m.group(2))
    sv = re.findall(r"saved (\S+result\.json)", f.read_text())
    if not sv:
        continue
    shutil.copy(sv[-1], d / "results" / f"{arm}-{rd}.json")
    r = json.load(open(sv[-1]))
    sp = r["speed"]
    data.setdefault(arm, {})[rd] = {c["streams"]: (c["aggregate_decode_tok_s"], c.get("overlap_decode_tok_s")) for c in sp["concurrency"]}
L = []
for idx, name in ((0, "aggregate_decode_tok_s (tokens / wall)"), (1, "overlap_decode_tok_s (all streams decoding)")):
    L.append(f"**{name}**\n")
    L.append("| streams | " + " | ".join(f"{x} median (rounds)" for x in sorted(data)) + f" | paired {a.num}/{a.den} per round | median ratio | min ratio |")
    L.append("|---|" + "---|" * len(data) + "---|---|---|")
    ns = sorted({n for x in data.values() for r in x.values() for n in r})
    for n in ns:
        cells = []
        for x in sorted(data):
            v = [data[x][r][n][idx] for r in sorted(data[x]) if n in data[x][r] and data[x][r][n][idx]]
            cells.append(f"{st.median(v):.1f} (" + " / ".join(f"{y:.1f}" for y in v) + ")" if v else "-")
        rat = []
        for r in sorted(set(data.get(a.num, {})) & set(data.get(a.den, {}))):
            p, q = data[a.num][r].get(n, (0, 0))[idx], data[a.den][r].get(n, (0, 0))[idx]
            if p and q:
                rat.append(p / q)
        L.append(f"| {n} | " + " | ".join(cells) + " | " + " / ".join(f"{x:.3f}" for x in rat) + " | " +
                 (f"{st.median(rat):.3f}" if rat else "-") + " | " + (f"{min(rat):.3f}" if rat else "-") + " |")
    L.append("")
print("\n".join(L))
if a.md:
    Path(a.md).write_text("\n".join(L) + "\n")
