#!/usr/bin/env python3
"""Lane dsv41/pdl: the MoE chain (gate/up, epilogue, down, combine), from `nsys stats --report cuda_gpu_trace --format csv`.

Usage: pdlchain.py TRACE.csv [more.csv ...]

A chain runs from `group_kernel` to `combine_kernel`; in it the earliest `grouped_cp_kernel` is gate/up and the latest is down.
chain_us = combine start - gate/up start: the critical path of the pair, which does not depend on how long a kernel sits in
griddepcontrol.wait (summed kernel time does). lead_us = epilogue end - down start. Chains are classed by the epilogue's grid
x (rows x top-k). A pass ends at the logits all-gather; pass_ms = its first kernel's start to its last kernel's end, for passes
with at least 20 chains.
"""
import csv
import statistics as st
import sys
from collections import defaultdict


def load(path):
    with open(path, newline="") as f:
        lines = [l for l in f if l.strip()]
    start = next(i for i, l in enumerate(lines) if l.startswith("Start") or l.startswith('"Start'))
    rows = []
    for r in csv.DictReader(lines[start:]):
        if r.get("Name") and r.get("Duration (ns)"):
            rows.append((int(r["Start (ns)"]), int(r["Duration (ns)"]), r["Name"], int(r.get("GrdX") or 0)))
    rows.sort()
    return rows


def chains(rows):
    """[(start index in rows, grid x, gate/up, epilogue, down, combine)], each kernel a (start, end) in us."""
    out, cur = [], None
    for s, d, n, gx in rows:
        k = (s / 1e3, (s + d) / 1e3)
        if "group_kernel" in n:
            cur = {"g": [], "e": None, "gx": 0}
        elif cur is None:
            continue
        elif "grouped_cp_kernel" in n:
            cur["g"].append(k)
        elif "epilogue_kernel" in n:
            cur["e"], cur["gx"] = k, gx
        elif "combine_kernel" in n:
            if len(cur["g"]) == 2 and cur["e"]:
                out.append((cur["gx"], cur["g"][0], cur["e"], cur["g"][1], k))
            cur = None
    return out


def med(xs):
    return st.median(xs) if xs else float("nan")


def report(path):
    rows = load(path)
    cs = chains(rows)
    print(f"{path}: {len(rows)} trace rows, {len(cs)} MoE chains")
    by = defaultdict(list)
    for c in cs:
        by[c[0]].append(c)
    print("  grid_x chains gate_us epi_us down_us chain_us lead_us(p10/med/p90) lead>30us down_before_gate_end_us")
    for gx in sorted(by):
        v = by[gx]
        lead = sorted(e[1] - d[0] for _, _, e, d, _ in v)
        print(
            f"  {gx:6d} {len(v):6d} {med([g[1] - g[0] for _, g, _, _, _ in v]):7.1f} {med([e[1] - e[0] for _, _, e, _, _ in v]):6.1f}"
            f" {med([d[1] - d[0] for _, _, _, d, _ in v]):7.1f} {med([c[0] - g[0] for _, g, _, _, c in v]):8.1f}"
            f"   {lead[len(lead) // 10]:7.1f} {lead[len(lead) // 2]:7.1f} {lead[len(lead) * 9 // 10]:7.1f}"
            f"  {sum(x > 30 for x in lead) / len(lead):6.1%} {med([g[1] - d[0] for _, g, _, d, _ in v]):8.1f}"
        )
    # passes
    cur, ps = [], []
    for r in rows:
        cur.append(r)
        if "AllGather" in r[2]:
            ps.append(cur)
            cur = []
    agg = defaultdict(list)
    for p in ps:
        c = chains(p)
        if len(c) < 20:
            continue
        span = (max(s + d for s, d, _, _ in p) - p[0][0]) / 1e6
        agg[c[0][0]].append((span, sum(x[4][0] - x[1][0] for x in c) / 1e3, len(c)))
    if agg:
        print("  grid_x passes pass_ms moe_chain_ms_per_pass chains_per_pass")
        for gx in sorted(agg):
            v = agg[gx][2:]  # the first two of a class: warm-up
            if v:
                print(f"  {gx:6d} {len(v):6d} {med([x[0] for x in v]):7.2f} {med([x[1] for x in v]):7.2f} {med([x[2] for x in v]):6.0f}")


for f in sys.argv[1:]:
    report(f)
