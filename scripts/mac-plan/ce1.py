#!/usr/bin/env python3
"""Per-op decode table from a ce1 trace (one command buffer per dispatch). Exclusive time = time after the previous buffer ended (MDEC's rule).
usage: ce1.py trace.jsonl tokens"""
import json, sys, collections
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
ntok = int(sys.argv[2])
last_pre = max(i for i, r in enumerate(recs) if r["t"] >= 64)
rs = recs[last_pre + 1:]
# drop the first decode step's worth of warm-up: keep the last (ntok - 4) / ntok of the records
rs = rs[len(rs) * 4 // ntok:]; n = ntok - 4
t = collections.Counter(); c = collections.Counter(); b = collections.Counter(); end = rs[0]["s"]; idle = 0.0
for r in rs:
    s = max(r["s"], end); idle += max(0.0, r["s"] - end); t[r["op"]] += max(0.0, r["e"] - s); c[r["op"]] += r["nd"]; b[r["op"]] += r["bytes"]; end = max(end, r["e"])
tot = sum(t.values())
print(f"per token: {tot*1e3/n:.2f} ms busy, {idle*1e3/n:.2f} ms idle, {sum(c.values())/n:.0f} dispatches")
print("| op | launches / token | us / token | share | us / launch | MB / token | GB/s |"); print("|---|---:|---:|---:|---:|---:|---:|")
for o, v in t.most_common():
    mb = b[o] / n / 1e6
    print(f"| {o} | {c[o]/n:.1f} | {v*1e6/n:.0f} | {100*v/tot:.1f} % | {v*1e6/c[o]:.1f} | {mb:.0f} | {(b[o]/1e9)/v if v and b[o] else 0:.0f} |")
