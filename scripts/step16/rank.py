#!/usr/bin/env python3
"""Lane msd: rank the kernels of an N-stream burst from an nsys cuda_gpu_trace csv.

    rank.py trace_cuda_gpu_trace.csv [decode_from_s [decode_to_s]]
The burst is the last cluster of kernels after an idle gap >= 1 s. Prints the whole burst, then (if decode_from_s is given, in
seconds after the burst's first kernel) the decode-only window: kernels grouped by template-stripped name with ms, share, calls,
mean duration, and GPU idle share (window span minus the union of kernel intervals).
"""
import csv, re, sys


def key(name):
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
    ev.append((s, s + float(r["Duration (ns)"]), name))
ev.sort(key=lambda e: e[0])
clusters, cur, last_end = [], [], None
for e in ev:
    if last_end is not None and e[0] - last_end >= 1e9:
        clusters.append(cur); cur = []
    cur.append(e); last_end = max(last_end or 0, e[1])
clusters.append(cur)
q = clusters[-1]
t0 = q[0][0]


def report(title, evs, lo, hi):
    groups = {}
    for s, e, n in evs:
        g = groups.setdefault(key(n), [0.0, 0]); g[0] += e - s; g[1] += 1
    tot = sum(g[0] for g in groups.values())
    busy, cs, ce = 0.0, None, None
    for s, e, _ in sorted(evs):
        if ce is None or s > ce:
            if ce is not None: busy += ce - cs
            cs, ce = s, e
        else: ce = max(ce, e)
    if ce is not None: busy += ce - cs
    span = hi - lo
    print(f"-- {title}: span {span / 1e6:.0f} ms, {len(evs)} kernels, kernel sum {tot / 1e6:.0f} ms, union busy {busy / 1e6:.0f} ms, GPU idle {100 * (1 - busy / span):.1f} %")
    for k, (t, c) in sorted(groups.items(), key=lambda x: -x[1][0])[:28]:
        print(f"{k:<82} {t / 1e6:8.1f} ms {100 * t / tot:5.1f} % {c:6d} calls {t / c / 1e3:9.1f} us avg")


print(f"clusters: {[(len(c), round((c[-1][1] - c[0][0]) / 1e9, 1)) for c in clusters if len(c) > 50]}")
report("whole burst", q, t0, max(e[1] for e in q))
if len(sys.argv) > 2:
    lo = t0 + float(sys.argv[2]) * 1e9
    hi = t0 + float(sys.argv[3]) * 1e9 if len(sys.argv) > 3 else max(e[1] for e in q) - 0.3e9
    # kernels whose start is inside the window
    w = [e for e in q if lo <= e[0] <= hi]
    report(f"decode window {float(sys.argv[2]):.1f}s..{(hi - t0) / 1e9:.1f}s", w, lo, hi)
