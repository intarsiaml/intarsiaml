#!/usr/bin/env python3
"""Lane dsv41/graphs: kernel time per pass kind, from `nsys stats --report cuda_gpu_trace --format csv` of a rank.

Usage: nsys_passes.py TRACE.csv [--ref DECODE_TRACE.csv]

Splits the kernel stream at the logits all-gather (one per pass, as nsys_breakdown.py) and classes each pass by its op
count: "big" passes (the verify or decode pass) and "small" ones (the DSpark block). The first two passes of each class are
dropped (warm-up, prompt). With --ref, a per-kernel delta against the big passes of a spec-off trace: what extra rows cost.
"""
import statistics as st
import sys
from collections import defaultdict

from nsys_breakdown import load, short


def passes(path):
    rows = load(path)
    out, cur = [], []
    for r in rows:
        cur.append(r)
        if "AllGather" in r[2]:
            out.append(cur)
            cur = []
    return out


def summary(ps):
    per = defaultdict(lambda: [0.0, 0.0])
    spans, busy = [], []
    for p in ps:
        spans.append((max(s + d for s, d, _, _ in p) - p[0][0]) / 1e6)
        iv, b, end = sorted((s, s + d) for s, d, _, _ in p), 0, 0
        for a, e in iv:
            b += max(0, e - max(a, end))
            end = max(end, e)
        busy.append(b / 1e6)
        for _, d, n, _ in p:
            k = short(n)
            per[k][0] += 1
            per[k][1] += d / 1e6
    n = len(ps)
    return n, st.median(spans), st.median(busy), {k: (v[0] / n, v[1] / n) for k, v in per.items()}


def classes(path):
    ps = [p for p in passes(path) if len(p) < 20000]
    cut = max(len(p) for p in ps) * 0.6
    big = [p for p in ps if len(p) >= cut]
    small = [p for p in ps if len(p) < cut]
    return big[2:], small[2:]


def show(name, s, ref=None, top=16):
    n, span, busy, per = s
    print(f"{name}: {n} passes, GPU span {span:.2f} ms, busy {busy:.2f} ms, kernel sum {sum(v[1] for v in per.values()):.2f} ms")
    keys = sorted(per, key=lambda k: -per[k][1])[:top]
    for k in keys:
        line = f"  {k:32s} {per[k][0]:7.1f} calls {per[k][1]:8.3f} ms"
        if ref:
            r = ref[3].get(k, (0.0, 0.0))
            line += f"   ref {r[0]:7.1f} calls {r[1]:8.3f} ms   delta {per[k][1] - r[1]:+8.3f} ms"
        print(line)


def main():
    a = sys.argv[1:]
    big, small = classes(a[0])
    ref = None
    if "--ref" in a:
        ref = summary(classes(a[a.index("--ref") + 1])[0])
        show("reference big passes", ref, top=8)
    show("big passes", summary(big), ref)
    if small:
        show("small passes", summary(small), top=8)


if __name__ == "__main__":
    main()
