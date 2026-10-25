#!/usr/bin/env python3
"""Lane wg: table of the served window (served.sh OUT dir). Per arm and stream count, the median over passes of
aggregate decode tok/s (localeval wall, includes the burst prefill), per-stream decode and per-stream PEAK decode tok/s,
steady ITL = 1000 / peak, steady aggregate = streams x peak. Then opt1 vs off (and vs main when present).
usage: summarize.py OUT_DIR"""
import json, glob, os, statistics, sys, collections

d = sys.argv[1]
rows = collections.defaultdict(list)   # (arm, streams) -> [(agg, per-stream, peak)]
for f in sorted(glob.glob(os.path.join(d, "*.result.json"))):
    arm = os.path.basename(f).split("-p")[0]
    sp = json.load(open(f)).get("speed", {})
    for r in sp.get("rows", []):
        if r.get("target_prompt_tokens") == 1024 and not r.get("errors"):
            rows[(arm, 1)].append((r["decode_tok_s"]["median"], r["decode_tok_s"]["median"], r["peak_decode_tok_s"]["median"]))
    for c in sp.get("concurrency", []):
        if not c.get("errors"):
            rows[(arm, c["streams"])].append((c["aggregate_decode_tok_s"], c["per_stream_decode_tok_s"]["median"],
                                              c["per_stream_peak_decode_tok_s"]["median"]))
med = statistics.median
tab = {}
print(f"{'arm':6}{'streams':>8}{'n':>3}{'agg tok/s':>11}{'per-stream':>11}{'peak/stream':>12}{'ITL ms':>8}{'steady agg':>11}   passes (agg)")
for (arm, s), v in sorted(rows.items(), key=lambda x: (x[0][1], x[0][0])):
    agg, per, peak = med([x[0] for x in v]), med([x[1] for x in v]), med([x[2] for x in v])
    tab[(arm, s)] = (agg, 1000 / peak, s * peak)
    print(f"{arm:6}{s:8d}{len(v):3d}{agg:11.1f}{per:11.1f}{peak:12.1f}{1000 / peak:8.2f}{s * peak:11.1f}   {[round(x[0], 1) for x in v]}")
for base in ("off", "main"):
    for s in sorted({k[1] for k in tab}):
        if ("opt1", s) in tab and (base, s) in tab:
            a, b = tab[("opt1", s)], tab[(base, s)]
            print(f"opt1 vs {base} @{s:2d} streams: agg {100 * (a[0] / b[0] - 1):+6.1f}%  steady ITL {100 * (a[1] / b[1] - 1):+6.1f}%  steady agg {100 * (a[2] / b[2] - 1):+6.1f}%")
