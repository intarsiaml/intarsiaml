#!/usr/bin/env python3
"""Lane dsv41/rdma: what the NCCL collectives of a decode / verify pass cost, from
`nsys stats --report cuda_gpu_trace --format csv` of ranks 0 and 1 (graph replay traced with --cuda-graph-trace=node).

Usage: nccl_cost.py R0.csv R1.csv [--passes N]

Passes are cut at the logits all-gather (as nsys_passes.py); big passes (>= 60 % of the largest kernel count) only, the
last N. Per pass and rank: the collective count, the kernel-duration sum, the wait before each one (its start minus the end
of the latest kernel that ended before it on any stream = how long the collective sat on a rank that had nothing else
to do), and, across the two ranks, the start skew of the k-th collective (rank 1 start - rank 0 start): a faster
transport cannot remove skew, only the part of the duration that is wire + launch.
"""
import csv
import statistics as st
import sys


def load(path):
    with open(path, newline="") as f:
        lines = [l for l in f if l.strip()]
    i = next(k for k, l in enumerate(lines) if l.startswith("Start") or l.startswith('"Start'))
    rows = []
    for r in csv.DictReader(lines[i:]):
        n = r.get("Name") or ""
        if not n or not r.get("Duration (ns)") or n.startswith("[CUDA"):
            continue
        rows.append((int(r["Start (ns)"]), int(r["Duration (ns)"]), n, r.get("Strm") or ""))
    rows.sort()
    return rows


def passes(rows):
    out, cur = [], []
    for r in rows:
        cur.append(r)
        if "AllGather" in r[2]:
            out.append(cur)
            cur = []
    return out


def big(ps, n):
    ps = [p for p in ps if len(p) < 20000]
    cut = max(len(p) for p in ps) * 0.6
    return [p for p in ps if len(p) >= cut][-n:]


def colls(p):
    """[(start, dur, wait_before)] of the NCCL kernels of a pass; wait = gap to the latest earlier kernel end."""
    out, maxend = [], 0
    for s, d, n, _ in p:
        if "nccl" in n.lower():
            out.append((s, d, max(0, s - maxend) if maxend else 0, n))
        maxend = max(maxend, s + d)
    return out


def q(v, f):
    v = sorted(v)
    return v[min(len(v) - 1, int(len(v) * f))]


def main():
    a = sys.argv[1:]
    npass = int(a[a.index("--passes") + 1]) if "--passes" in a else 30
    r0, r1 = (big(passes(load(a[i])), npass) for i in (0, 1))
    print(f"big passes used: rank 0 {len(r0)}, rank 1 {len(r1)}; kernels a pass r0 {st.median(len(p) for p in r0):.0f}")
    per = {}
    for tag, ps in (("r0", r0), ("r1", r1)):
        cnt, sm, dur, wait = [], [], [], []
        for p in ps:
            c = colls(p)
            cnt.append(len(c))
            sm.append(sum(x[1] for x in c) / 1e6)
            dur += [x[1] / 1e3 for x in c if "AllReduce_Sum_f32" in x[3]]
            wait += [x[2] / 1e3 for x in c if "AllReduce_Sum_f32" in x[3]]
        per[tag] = (cnt, sm, dur, wait)
        print(f"{tag}: collectives a pass median {st.median(cnt)}, duration sum a pass median {st.median(sm):.2f} ms (min {min(sm):.2f}, max {max(sm):.2f})")
        print(f"    f32 all-reduce duration us: median {st.median(dur):.1f} p10 {q(dur,.1):.1f} p90 {q(dur,.9):.1f} p99 {q(dur,.99):.1f}; wait before it us: median {st.median(wait):.1f} p90 {q(wait,.9):.1f}")
    # pair the k-th collective of the same pass on the two ranks (passes paired from the end; same count on both)
    mins, gaps, kinds, spans = [], [], {}, []
    skew, wire = [], []
    for p0, p1 in zip(reversed(r0), reversed(r1)):
        c0, c1 = colls(p0), colls(p1)
        if len(c0) != len(c1):
            continue
        mins.append(sum(min(x[1], y[1]) for x, y in zip(c0, c1)) / 1e6)
        gaps.append(sum(abs(x[1] - y[1]) for x, y in zip(c0, c1)) / 1e6)
        spans.append((max(s + d for s, d, _, _ in p0) - p0[0][0]) / 1e6)
        for x, y in zip(c0, c1):
            k = ("f32 all-reduce" if "AllReduce_Sum_f32" in x[3] else "bf16 all-reduce" if "AllReduce_Sum_bf16" in x[3] else "all-gather")
            kinds.setdefault(k, []).append((min(x[1], y[1]) / 1e3, abs(x[1] - y[1]) / 1e3))
    print(f"{len(mins)} passes paired. Per pass (median): rank-0 GPU span {st.median(spans):.1f} ms; sum of min(duration over the ranks) {st.median(mins):.2f} ms; sum of |d0 - d1| (skew the early rank waited) {st.median(gaps):.2f} ms")
    for k, v in kinds.items():
        m = [a for a, _ in v]; g = [b for _, b in v]
        print(f"  {k}: {len(v) / len(mins):.1f} a pass; min-over-ranks duration us median {st.median(m):.1f} p10 {q(m, .1):.1f} p90 {q(m, .9):.1f}; |d0-d1| us median {st.median(g):.1f} p90 {q(g, .9):.1f}")

main()
