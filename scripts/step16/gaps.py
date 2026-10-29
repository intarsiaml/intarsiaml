#!/usr/bin/env python3
"""Lane msd: split the inter-step gaps of a decode window of an nsys cuda_gpu_trace csv.
  gaps.py trace.csv from_s to_s   A step = the run of GPU ops between two idle gaps >= 300 us; prints per-step kernel time, the gap to the
next step and what ran in it (memcpy bytes by kind)."""
import csv, sys, collections
rows = list(csv.DictReader(open(sys.argv[1])))
ev = []
for r in rows:
    try: ev.append((float(r["Start (ns)"]), float(r["Duration (ns)"]), r))
    except (KeyError, ValueError): pass
ev.sort(key=lambda e: e[0])
cl, cur, last = [], [], None
for e in ev:
    if last is not None and e[0] - last >= 1e9: cl.append(cur); cur = []
    cur.append(e); last = max(last or 0, e[0] + e[1])
cl.append(cur); q = cl[-1]; t0 = q[0][0]
lo, hi = t0 + float(sys.argv[2]) * 1e9, t0 + float(sys.argv[3]) * 1e9
q = [e for e in q if lo <= e[0] <= hi]
steps, cur, end = [], [], None
for e in q:
    if end is not None and e[0] - end >= 300e3: steps.append(cur); cur = []
    cur.append(e); end = max(end or 0, e[0] + e[1])
steps.append(cur)
def memcpy(r): return "memcpy" in (r.get("Name") or "").lower() or "Memcpy" in (r.get("Name") or "") or r.get("Bytes") not in (None, "", "-") and not (r.get("GrdX") or "").strip().isdigit()
tot_gap = tot_span = 0; nb = collections.Counter(); nk = collections.Counter(); gaps = []
for a, b in zip(steps, steps[1:]):
    s_end = max(e[0] + e[1] for e in a); gap = b[0][0] - s_end
    tot_gap += gap; tot_span += (b[0][0] - a[0][0])
    gaps.append(gap)
    for e in a + b:
        pass
for st in steps:
    for s, d, r in st:
        if not (r.get("GrdX") or "").strip().isdigit():
            nm = (r.get("Name") or "")[:40]; nb[nm] += float((r.get("Bytes (MB)") or "0").replace(",", "") or 0); nk[nm] += d
print(f"steps {len(steps)}  mean step span {tot_span/max(1,len(steps)-1)/1e6:.2f} ms  mean gap {tot_gap/max(1,len(steps)-1)/1e6:.2f} ms  ({100*tot_gap/max(1,tot_span):.1f} %)")
gaps.sort(); n = len(gaps)
print(f"gap us: p10 {gaps[n//10]/1e3:.0f} p50 {gaps[n//2]/1e3:.0f} p90 {gaps[9*n//10]/1e3:.0f} max {gaps[-1]/1e3:.0f}")
for nm in nb: print(f"non-kernel op {nm!r}: per step {nb[nm]/len(steps):.3f} MB, {nk[nm]/len(steps)/1e3:.1f} us GPU time")
