#!/usr/bin/env python3
"""Lane dsv41/tp2: where one decode step's time goes, from `nsys stats --report cuda_gpu_trace --format csv` of a rank.

Usage: nsys_breakdown.py TRACE.csv [--tokens N] [--skip-ms MS]

The capture holds one request (short prompt, N forced output tokens). The decode phase is taken as the last N
"steps" found by splitting the kernel stream at gaps longer than --split-us (default 2000 us: the host's sample and the
next pass's build sit between steps). Per step: wall span (first kernel start to last kernel end), GPU busy time (union of
kernel intervals), NCCL kernel time (which includes waiting for the other rank), idle gaps inside the span (launch
latency / host), and the top kernels by summed duration.
"""
import csv
import sys
from collections import defaultdict


def load(path):
    rows = []
    with open(path, newline="") as f:
        # nsys prints a few notice lines before the header
        lines = [l for l in f if l.strip()]
    start = next(i for i, l in enumerate(lines) if l.startswith("Start") or l.startswith('"Start'))
    for r in csv.DictReader(lines[start:]):
        name = r.get("Name") or ""
        if not name or r.get("Duration (ns)") in (None, ""):
            continue
        if name.startswith("[CUDA memcpy") or name.startswith("[CUDA memset"):
            kind = "memcpy"
        else:
            kind = "kernel"
        rows.append((int(r["Start (ns)"]), int(r["Duration (ns)"]), name, kind))
    rows.sort()
    return rows


def short(name):
    n = name.replace("<unnamed>::", "").replace("(anonymous namespace)::", "")
    if n.startswith("void "):
        n = n[5:]
    if n.startswith("["):
        return n
    # drop the argument list and template arguments (the first '(' or '<' after the name)
    for i, ch in enumerate(n):
        if ch in "(<":
            n = n[:i]
            break
    if "nccl" in n.lower():
        return "nccl:" + n.split("_")[1] if "_" in n else n
    return n.split("::")[-1]


def main():
    args = sys.argv[1:]
    path = args[0]
    tokens = int(args[args.index("--tokens") + 1]) if "--tokens" in args else 32
    split_us = float(args[args.index("--split-us") + 1]) if "--split-us" in args else 2000.0
    rows = load(path)
    steps, cur, last_end = [], [], None
    by_gather = any("AllGather" in n for _, _, n, _ in rows)
    for s, d, n, k in rows:
        if not by_gather and last_end is not None and (s - last_end) / 1e3 > split_us and cur:
            steps.append(cur)
            cur = []
        cur.append((s, d, n, k))
        last_end = max(last_end or 0, s + d)
        if by_gather and "AllGather" in n:
            # rank 0 downloads the logits right after the gather: keep that copy in this pass
            steps.append(cur)
            cur = []
    if cur and not by_gather:
        steps.append(cur)
    elif cur and steps and all(k == "memcpy" for _, _, _, k in cur):
        steps[-1].extend(cur)
    how = "the logits all-gather (one per pass)" if by_gather else f"gaps > {split_us:.0f} us"
    print(f"{len(rows)} GPU ops, {len(steps)} passes split at {how}")
    dec = steps[-tokens:] if len(steps) >= tokens else steps
    # time between consecutive steps' starts = the real per-token time
    starts = [st[0][0] for st in dec]
    periods = [(b - a) / 1e3 for a, b in zip(starts, starts[1:])]
    agg = defaultdict(float)
    per_kernel = defaultdict(lambda: [0, 0.0])
    for st in dec:
        span = (max(s + d for s, d, _, _ in st) - st[0][0]) / 1e3
        busy, end = 0.0, None
        for s, d, n, k in st:
            e = s + d
            if end is None or s >= end:
                busy += d
                end = e
            elif e > end:
                busy += e - end
                end = e
            sn = short(n)
            per_kernel[sn][0] += 1
            per_kernel[sn][1] += d / 1e3
            if sn.startswith("nccl"):
                agg["nccl_us"] += d / 1e3
            elif k == "memcpy":
                agg["memcpy_us"] += d / 1e3
            else:
                agg["kernel_us"] += d / 1e3
        agg["span_us"] += span
        agg["busy_us"] += busy / 1e3
        agg["ops"] += len(st)
    n = len(dec)
    m = {k: v / n for k, v in agg.items()}
    per_tok = sum(periods) / len(periods) if periods else m["span_us"]
    print(f"decode steps analysed: {n}")
    print(f"  per token (start to start)      {per_tok/1e3:8.2f} ms  -> {1e6/per_tok:.1f} tok/s")
    print(f"  GPU span of a step               {m['span_us']/1e3:8.2f} ms   ({m['ops']:.0f} GPU ops)")
    print(f"  GPU busy (union)                 {m['busy_us']/1e3:8.2f} ms")
    print(f"    compute kernels (sum)          {m['kernel_us']/1e3:8.2f} ms")
    print(f"    NCCL kernels (sum, incl. wait) {m['nccl_us']/1e3:8.2f} ms")
    print(f"    memcpy / memset (sum)          {m.get('memcpy_us', 0)/1e3:8.2f} ms")
    print(f"  idle gaps inside the span        {(m['span_us']-m['busy_us'])/1e3:8.2f} ms   (launch latency, host waits)")
    print(f"  between steps (host: sample, build, Engram, logits copy) {(per_tok-m['span_us'])/1e3:8.2f} ms")
    print("  top kernels per step:")
    for k, (c, us) in sorted(per_kernel.items(), key=lambda kv: -kv[1][1])[:18]:
        print(f"    {k:48s} {c/n:7.1f} calls {us/n/1e3:8.3f} ms  ({us/n/per_tok*100:5.1f} %)")


if __name__ == "__main__":
    main()
