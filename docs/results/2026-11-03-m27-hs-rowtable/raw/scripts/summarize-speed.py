#!/usr/bin/env python3
"""Collect the ABAB speed cells from localeval result dirs (labels m27rt-<arm>-mb<MB>-<n>-<size>) into raw/speed-table.md and raw/speed-cells.json."""
import json, glob, os, re, statistics as st, sys
out = sys.argv[1]
cells = []
for d in sorted(glob.glob("/opt/code/deepeval/results/*-adhoc-m27rt-*")):
    m = re.search(r"m27rt-(def|rt)-mb(\d+)-(\d+)-(\d+)$", d)
    if not m:
        continue
    arm, mb, n, size = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    r = json.load(open(d + "/result.json"))
    sp = r["speed"]
    single = sp["rows"][0]["decode_tok_s"]["median"] if sp["rows"] else None
    for c in sp["concurrency"]:
        cells.append(dict(arm=arm, mb=mb, run=n, size=size, streams=c["streams"], agg=c["aggregate_decode_tok_s"],
                          per_stream=c["per_stream_decode_tok_s"]["median"], per_stream_peak=c["per_stream_peak_decode_tok_s"]["median"],
                          ttft_max=c["ttft_s"]["max"], errors=len(c["errors"]), single=single, dir=os.path.basename(d)))
json.dump(cells, open(out + "/speed-cells.json", "w"), indent=1)
def spread(v): return (max(v) - min(v)) / st.median(v) * 100 if len(v) > 1 else 0.0
lines = ["| max-batch | prompt | streams | arm | cells (agg tok/s, in order) | median agg | spread % | median per-stream | median per-stream peak |", "|---|---|---|---|---|---|---|---|---|"]
groups = {}
for c in cells:
    groups.setdefault((c["mb"], c["size"], c["streams"], c["arm"]), []).append(c)
ratios = []
for (mb, size, streams) in sorted({k[:3] for k in groups}):
    row = {}
    for arm in ("def", "rt"):
        g = sorted(groups.get((mb, size, streams, arm), []), key=lambda c: (c["run"]))
        if not g: continue
        a = [c["agg"] for c in g]
        row[arm] = a
        lines.append(f"| {mb} | {size} | {streams} | {'A baseline' if arm=='def' else 'B rowtab-HSB'} | {', '.join(f'{x:.1f}' for x in a)} | {st.median(a):.1f} | {spread(a):.1f} | {st.median(c['per_stream'] for c in g):.2f} | {st.median(c['per_stream_peak'] for c in g):.1f} |")
    if "def" in row and "rt" in row:
        ratios.append((mb, size, streams, st.median(row["rt"]) / st.median(row["def"])))
lines += ["", "| max-batch | prompt | streams | B / A (median aggregate) |", "|---|---|---|---|"] + [f"| {mb} | {s} | {n} | x{r:.3f} ({(r-1)*100:+.1f} %) |" for mb, s, n, r in ratios]
open(out + "/speed-table.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
