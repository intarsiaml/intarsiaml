#!/usr/bin/env python3
"""Lane dsv41/prefill: per-kernel-class time of prefill requests, from `nsys stats --report cuda_gpu_trace --format csv`.

Usage: prefill_table.py TRACE.csv [--gap-ms 1000] [--top 30] [--classes ours|tf] [--chunks N,N,...]

The capture holds one or more prefill requests (max_tokens 1) separated by idle gaps of at least --gap-ms. Per request:
span (first kernel start to last kernel end), GPU busy (union of kernel intervals), and per kernel name the call count and
two times:
  sum  = summed kernel durations
  excl = wall time attributed to the kernel that started last among those running (a kernel that waits inside a PDL
         griddepcontrol.wait started earlier than the kernel it waits for, so it is not charged for the wait)
`excl` adds up to the busy time; `sum` can exceed it where kernels overlap. Kernel names are grouped into classes by
CLASSES (first matching regex); --chunks gives the chunk count of each request (in file order) for the per-chunk columns.
"""
import argparse
import csv
import re
import sys
from collections import defaultdict


def load(path):
    with open(path, newline="") as f:
        lines = [l for l in f if l.strip()]
    start = next(i for i, l in enumerate(lines) if l.startswith("Start") or l.startswith('"Start'))
    rows = []
    for r in csv.DictReader(lines[start:]):
        name = r.get("Name") or ""
        if not name or r.get("Duration (ns)") in (None, ""):
            continue
        rows.append((int(r["Start (ns)"]), int(r["Duration (ns)"]), name, r.get("Strm", "")))
    rows.sort()
    return rows


def short(name):
    n = name.replace("<unnamed>::", "").replace("(anonymous namespace)::", "")
    if n.startswith("void "):
        n = n[5:]
    n = re.sub(r"\(.*$", "", n) if "(" in n and not n.startswith("[") else n
    return n[:90]


def excl_times(rows):
    """Wall time attributed to the latest-started running kernel (events sweep). Returns {index: ns}."""
    ev = []
    for i, (s, d, _, _) in enumerate(rows):
        ev.append((s, 1, i))
        ev.append((s + d, 0, i))
    ev.sort()
    out = defaultdict(int)
    active = []  # (start, idx), kept in start order
    last = None
    for t, kind, i in ev:
        if last is not None and active and t > last:
            out[active[-1][1]] += t - last
        if kind == 1:
            active.append((rows[i][0], i))
            active.sort()
        else:
            active = [a for a in active if a[1] != i]
        last = t
    return out


def split(rows, gap_ns):
    reqs, cur, end = [], [], 0
    for r in rows:
        if cur and r[0] - end > gap_ns:
            reqs.append(cur)
            cur, end = [], 0
        cur.append(r)
        end = max(end, r[0] + r[1])
    if cur:
        reqs.append(cur)
    return reqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--gap-ms", type=float, default=1000)
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--chunks", default="")
    ap.add_argument("--tokens", default="")
    ap.add_argument("--min-ms", type=float, default=50, help="requests shorter than this are dropped (warm-ups, noise)")
    a = ap.parse_args()
    rows = load(a.trace)
    reqs = [q for q in split(rows, int(a.gap_ms * 1e6)) if (max(r[0] + r[1] for r in q) - q[0][0]) > a.min_ms * 1e6]
    chunks = [int(x) for x in a.chunks.split(",")] if a.chunks else [1] * len(reqs)
    toks = [int(x) for x in a.tokens.split(",")] if a.tokens else [0] * len(reqs)
    print(f"{len(rows)} kernels, {len(reqs)} requests (gap >= {a.gap_ms} ms, >= {a.min_ms} ms long)")
    for qi, q in enumerate(reqs):
        t0 = q[0][0]
        span = max(r[0] + r[1] for r in q) - t0
        ex = excl_times(q)
        busy = sum(ex.values())
        nch = chunks[qi] if qi < len(chunks) else 1
        nt = toks[qi] if qi < len(toks) else 0
        print(f"\n=== request {qi}: {len(q)} kernels, span {span / 1e6:.1f} ms, busy {busy / 1e6:.1f} ms ({100 * busy / span:.1f} %), "
              f"{nch} chunk(s), {nt} tokens ===")
        by = defaultdict(lambda: [0, 0, 0])
        for i, (s, d, n, _) in enumerate(q):
            k = short(n)
            by[k][0] += 1
            by[k][1] += d
            by[k][2] += ex.get(i, 0)
        tot_ex = sum(v[2] for v in by.values())
        print(f"{'kernel':<92}{'calls':>7}{'sum ms':>10}{'excl ms':>10}{'excl %':>8}{'excl ms/chunk':>15}{'excl us/token':>15}")
        for k, v in sorted(by.items(), key=lambda kv: -kv[1][2])[: a.top]:
            print(f"{k:<92}{v[0]:>7}{v[1] / 1e6:>10.2f}{v[2] / 1e6:>10.2f}{100 * v[2] / tot_ex:>8.1f}{v[2] / 1e6 / nch:>15.2f}{(v[2] / 1e3 / nt) if nt else 0:>15.1f}")


if __name__ == "__main__":
    main()
