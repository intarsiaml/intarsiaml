#!/usr/bin/env python3
"""herdflip: markdown TTFT table (sum / p50 / last of the 8 requests per burst cell, seconds) from herd.jsonl (herd_bench.py or burst.py rows). usage: table.py herd.jsonl"""
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
cells = sorted({(r["seed"], r.get("stagger", r.get("gap", 0))) for r in rows if r["mode"] != "solo"}, key=lambda c: c[0])
arms = list(dict.fromkeys((r.get("visit") or r["arm"]) for r in rows))
print("| arm | " + " | ".join(f"stagger {c[1]} s: sum / p50 / last" for c in cells) + " |")
print("|---|" + "---|" * len(cells))
for a in arms:
    out = []
    for s, g in cells:
        r = next((x for x in rows if (x.get("visit") or x["arm"]) == a and x["seed"] == s and x["mode"] != "solo"), None)
        t = [x for x in (r.get("ttft_s") or r.get("ttft") or []) if x is not None] if r else []
        out.append(f"{sum(t):.1f} / {statistics.median(t):.2f} / {max(t):.1f}" if t else "-")
    print(f"| `{a}` | " + " | ".join(out) + " |")
