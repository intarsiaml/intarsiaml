#!/usr/bin/env python3
"""Lane spf: split an nsys cuda_gpu_trace into requests (idle gap >= 100 ms) and rank the kernels of the last one.

    rank_req.py trace_cuda_gpu_trace.csv [request-index]
Prints, per request: kernels, span, busy; for the chosen request (default last): kernels grouped by template-stripped name
with ms, share, calls, mean duration, plus the launch-gap total.
"""
import csv
import re
import sys


def key(name: str) -> str:
    n = name.replace("void ", "")
    n = re.sub(r"^(\(anonymous namespace\)|<unnamed>)::", "", n)
    n = re.sub(r"\(.*", "", n)
    n = re.sub(r"<\(int\)", "<", n).replace("(bool)", "")
    return n[:80]


ev = []
for r in csv.DictReader(open(sys.argv[1])):
    name = r.get("Name") or ""
    if name.startswith("[CUDA"):
        continue
    s = float(r["Start (ns)"])
    ev.append((s, s + float(r["Duration (ns)"]), name, r))
ev.sort(key=lambda e: e[0])
reqs, cur, last_end = [], [], None
for e in ev:
    if last_end is not None and e[0] - last_end >= 100e6:
        reqs.append(cur); cur = []
    cur.append(e); last_end = max(last_end or 0, e[1])
reqs.append(cur)
for i, q in enumerate(reqs):
    t0, t1 = q[0][0], max(e[1] for e in q)
    print(f"request {i}: {len(q)} kernels, span {(t1 - t0) / 1e6:.1f} ms, kernel time {sum(e[1] - e[0] for e in q) / 1e6:.1f} ms")
idx = int(sys.argv[2]) if len(sys.argv) > 2 else len(reqs) - 1
q = reqs[idx]
groups = {}
for s, e, n, _ in q:
    g = groups.setdefault(key(n), [0.0, 0])
    g[0] += e - s; g[1] += 1
tot = sum(g[0] for g in groups.values())
print(f"-- request {idx}: kernel total {tot / 1e6:.1f} ms")
for k, (t, c) in sorted(groups.items(), key=lambda x: -x[1][0])[:22]:
    print(f"{k:<82} {t / 1e6:8.1f} ms {100 * t / tot:5.1f} % {c:6d} calls {t / c / 1e3:9.1f} us avg")
busy, cs, ce, gaps = 0.0, q[0][0], q[0][1], []
for s, e, n, _ in q[1:]:
    if s > ce:
        busy += ce - cs; gaps.append(s - ce); cs, ce = s, e
    else:
        ce = max(ce, e)
busy += ce - cs
print(f"span {(ce - q[0][0]) / 1e6:.1f} ms busy {busy / 1e6:.1f} ms idle {(ce - q[0][0] - busy) / 1e6:.2f} ms in {len(gaps)} gaps; largest {max(gaps, default=0) / 1e3:.0f} us")
# GEMM launches by kernel name and grid, first layer of the request (shape evidence)
