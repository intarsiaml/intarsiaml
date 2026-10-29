#!/usr/bin/env python3
"""Lane msd: per (kernel, grid, block) calls and mean duration inside a decode window of an nsys cuda_gpu_trace csv.
  shapes.py trace.csv from_s to_s [name-regex]   (seconds after the burst's first kernel, as rank.py)"""
import csv, re, sys
rows = list(csv.DictReader(open(sys.argv[1])))
ev = [(float(r["Start (ns)"]), float(r["Duration (ns)"]), r) for r in rows if not (r.get("Name") or "").startswith("[CUDA")]
ev.sort(key=lambda e: e[0])
clusters, cur, last = [], [], None
for e in ev:
    if last is not None and e[0] - last >= 1e9:
        clusters.append(cur); cur = []
    cur.append(e); last = max(last or 0, e[0] + e[1])
clusters.append(cur)
q = clusters[-1]; t0 = q[0][0]
lo, hi = t0 + float(sys.argv[2]) * 1e9, t0 + float(sys.argv[3]) * 1e9
pat = re.compile(sys.argv[4]) if len(sys.argv) > 4 else None
g = {}
for s, d, r in q:
    if not lo <= s <= hi: continue
    n = re.sub(r"\(.*", "", r["Name"].replace("void ", ""))[:60]
    if pat and not pat.search(n): continue
    k = (n, f'{r["GrdX"]}x{r["GrdY"]}x{r["GrdZ"]}', f'{r["BlkX"]}x{r["BlkY"]}')
    a = g.setdefault(k, [0, 0.0]); a[0] += 1; a[1] += d
steps = sum(1 for s, d, r in q if lo <= s <= hi and "embedding_kernel" in r["Name"]) or 1
print(f"steps in window: {steps}")
for k, (c, t) in sorted(g.items(), key=lambda x: -x[1][1])[:int(sys.argv[5]) if len(sys.argv) > 5 else 30]:
    print(f"{k[0]:<48} grid {k[1]:<12} blk {k[2]:<7} {c/steps:6.1f} calls/step {t/c/1e3:8.1f} us  {t/steps/1e6:6.2f} ms/step")
