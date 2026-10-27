#!/usr/bin/env python3
"""Per-op table of an INTARSIA_METAL_PROFILE=kernel trace of one prefill (lane M35P). Usage: perop.py kernel.jsonl [cb.jsonl]
Records with t < 16 tokens (the 2-token decode steps) are dropped; the rest is the prefill."""
import json, sys, collections
rs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
pf = [r for r in rs if r["t"] >= 16]
agg = collections.defaultdict(lambda: [0, 0.0])
tot = 0.0
for r in pf:
    d = r["e"] - r["s"]; tot += d
    a = agg[r["op"]]; a[0] += 1; a[1] += d
print(f"prefill kernel-mode GPU sum {tot*1e3:.1f} ms over {len(pf)} launches (all {len(rs)})")
print(f"{'op':44} {'calls':>6} {'ms':>9} {'%':>6} {'us/call':>9}")
for op, (c, d) in sorted(agg.items(), key=lambda x: -x[1][1]):
    print(f"{op:44} {c:6} {d*1e3:9.1f} {100*d/tot:6.1f} {d/c*1e6:9.0f}")
if len(sys.argv) > 2:
    cb = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
    big = [c for c in cb if c.get("nd", 0) > 50]
    busy = sum(c["e"] - c["s"] for c in cb); span = cb[-1]["e"] - cb[0]["s"]
    print(f"cb mode: {len(cb)} command buffers, GPU busy {busy*1e3:.1f} ms of span {span*1e3:.1f} ms ({100*busy/span:.1f}%)")
