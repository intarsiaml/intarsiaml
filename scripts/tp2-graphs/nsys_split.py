#!/usr/bin/env python3
"""Lane G1c: split the tp=2 decode step's all-reduce time into wait and transfer from two nsys traces (one per rank).

    nsys_split.py <rank0 cuda_gpu_trace.csv> <rank1 cuda_gpu_trace.csv> [--steps N]

Both ranks issue the same collectives in the same order, so the i-th NCCL kernel on rank 0 and the i-th on rank 1 are
the same all-reduce. A collective's kernel on a rank runs from that rank's launch until both ranks' data has crossed,
so per call: transfer ~= min(d0, d1) (the rank that arrived last waits for nothing) and wait_r = d_r - min(d0, d1)
(time rank r sat in the kernel for the other rank). No clock sync between the hosts is needed.

Decode steps are delimited on rank 0 by the argmax kernel (one per step, the head's last op); the last N steps are
used (default 48: skips capture and warm-up). Per step on rank 0: wall (argmax to argmax), busy kernel time, NCCL
time, idle gaps (no kernel running: launch / host / replay overhead).
"""
import csv
import statistics as st
import sys


def load(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            name = r.get("Name") or r.get("name") or ""
            try:
                s = int(float(r.get("Start (ns)") or r.get("Start") or 0))
                d = int(float(r.get("Duration (ns)") or r.get("Duration") or 0))
            except ValueError:
                continue
            # kernels only (memcpy / memset rows carry other names in [brackets])
            if name.startswith("["):
                continue
            rows.append((s, d, name))
    rows.sort()
    return rows


def is_nccl(n):
    return n.startswith("ncclDevKernel") or n.startswith("ncclKernel") or "nccl" in n.lower()


def main():
    a = sys.argv[1:]
    steps = 48
    if "--steps" in a:
        i = a.index("--steps")
        steps = int(a[i + 1])
        del a[i : i + 2]
    r0, r1 = load(a[0]), load(a[1])
    n0 = [(s, d) for s, d, n in r0 if is_nccl(n)]
    n1 = [(s, d) for s, d, n in r1 if is_nccl(n)]
    print(f"nccl kernels: rank0 {len(n0)}, rank1 {len(n1)}")
    if len(n0) != len(n1):
        print("WARNING: different NCCL kernel counts; pairing by order from the end")
    k = min(len(n0), len(n1))
    n0, n1 = n0[-k:], n1[-k:]
    # rank-0 decode steps from argmax kernels
    am = [s for s, d, n in r0 if "argmax" in n.lower()]
    if len(am) < steps + 1:
        steps = max(0, len(am) - 1)
    am = am[-(steps + 1) :]
    t_lo, t_hi = am[0], am[-1]
    pairs = [(x, y) for x, y in zip(n0, n1) if t_lo <= x[0] < t_hi]
    if not pairs:
        print("no NCCL kernels inside the decode window")
        return
    d0 = [x[1] / 1e3 for x, _ in pairs]
    d1 = [y[1] / 1e3 for _, y in pairs]
    tr = [min(p, q) for p, q in zip(d0, d1)]
    w0 = [p - t for p, t in zip(d0, tr)]
    w1 = [q - t for q, t in zip(d1, tr)]
    per = len(pairs) / steps
    print(f"decode window: {steps} steps, {len(pairs)} all-reduces ({per:.1f} per step)")
    f = lambda v: f"mean {st.mean(v):7.1f}  median {st.median(v):7.1f}  p90 {sorted(v)[int(0.9 * (len(v) - 1))]:7.1f}"
    print(f"  per call us  rank0 kernel  {f(d0)}")
    print(f"               rank1 kernel  {f(d1)}")
    print(f"               transfer=min  {f(tr)}")
    print(f"               wait rank0    {f(w0)}")
    print(f"               wait rank1    {f(w1)}")
    # per-step rank-0 budget
    walls, busy, nccl, gaps = [], [], [], []
    for lo, hi in zip(am[:-1], am[1:]):
        ks = [(s, d, n) for s, d, n in r0 if lo <= s < hi]
        walls.append((hi - lo) / 1e3)
        busy.append(sum(d for _, d, _ in ks) / 1e3)
        nccl.append(sum(d for _, d, n in ks if is_nccl(n)) / 1e3)
        g, end = 0, lo
        for s, d, _ in ks:
            if s > end:
                g += s - end
            end = max(end, s + d)
        gaps.append(g / 1e3)
    m = st.mean
    tr_step, w0_step = m(tr) * per, m(w0) * per
    print(f"rank-0 step (us, mean over {steps}): wall {m(walls):.0f} = kernels {m(busy):.0f} + idle gaps {m(gaps):.0f}")
    print(f"  of the kernels: NCCL {m(nccl):.0f} (transfer {tr_step:.0f} + rank-0 wait {w0_step:.0f}), compute {m(busy) - m(nccl):.0f}")


if __name__ == "__main__":
    main()
