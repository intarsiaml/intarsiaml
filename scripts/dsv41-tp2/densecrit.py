#!/usr/bin/env python3
"""Lane dsv41/group: launches per pass and the dense critical path of a decode / verify pass, from
`nsys stats --report cuda_gpu_trace --format csv` of rank 0 (graph replay traced with --cuda-graph-trace=node, lanes on).

Usage: densecrit.py TRACE.csv [--passes N] [--min-kernels K]

Passes are cut at the logits all-gather (as nsys_passes.py). Only "big" passes (at least --min-kernels kernels: the
decode or verify pass, not a DSpark draft block) of the last --passes are used. For each class of kernel:
  calls   launches per pass
  sum     kernel-duration sum per pass (ms), every stream
  crit    wall time per pass charged to the class of the kernel that STARTED LAST among those running (the
          draftdev/crit.py rule: a kernel launched on top of another takes the time, side-stream work behind
          the pass stream's front costs nothing)
  s0      sum on the pass stream (the stream of the pass's first kernel)
Then the EXL3 dense shapes: per (kernel, K2, grid) calls, median duration, and the helper kernels around them.
"""
import csv
import statistics as st
import sys
from collections import defaultdict


def load(path):
    with open(path, newline="") as f:
        lines = [l for l in f if l.strip()]
    i = next(k for k, l in enumerate(lines) if l.startswith("Start") or l.startswith('"Start'))
    rows = []
    for r in csv.DictReader(lines[i:]):
        n = r.get("Name") or ""
        if not n or not r.get("Duration (ns)"):
            continue
        if n.startswith("[CUDA"):
            continue
        rows.append((int(r["Start (ns)"]), int(r["Duration (ns)"]), n, r.get("GrdX") or "", r.get("GrdY") or "", r.get("Strm") or ""))
    rows.sort()
    return rows


def short(n):
    n = n.replace("<unnamed>::", "").replace("(anonymous namespace)::", "").replace("void ", "")
    for i, ch in enumerate(n):
        if ch in "(":
            n = n[:i]
            break
    return n


def klass(n):
    s = short(n)
    if "nccl" in s.lower():
        return "nccl"
    if s.startswith("rot_in_kernel<"):
        return "moe.rot_in"
    if s.startswith("rot_in_kernel"):
        return "dense.rot_in"
    if s.startswith(("split_kernel", "linear_kernel", "gemm_kernel")):
        return "dense.main"
    if s.startswith("split_finish_kernel"):
        return "dense.finish"
    if s.startswith("dense_f16w"):
        return "dense.f16"
    if s.startswith("transpose01"):
        return "dense.transpose"
    if s.startswith(("qdq", "ik_f32_to_bf16")):
        return "dense.cast"
    if "grouped" in s or "epilogue_kernel" in s or "combine_kernel" in s or "group_kernel" in s or "moe_route" in s:
        return "moe"
    if s.startswith(("hc_", "hc")):
        return "hc"
    if s.startswith(("attn", "index", "window_build", "ring_write", "rope_tail", "compress", "cmp", "topk", "select")):
        return "attn"
    if "prefetch" in s.lower() or s.startswith("l2_"):
        return "prefetch"
    return "other"


def passes(rows):
    out, cur = [], []
    for r in rows:
        cur.append(r)
        if "AllGather" in r[2]:
            out.append(cur)
            cur = []
    return out


def partition(p, cf):
    xs = [(a, a + d, cf(n)) for a, d, n, *_ in p if cf(n) != "prefetch"]
    pts = sorted({t for a, b, c in xs for t in (a, b)})
    xs.sort()
    out = defaultdict(float)
    active, i = [], 0
    for lo, hi in zip(pts, pts[1:]):
        while i < len(xs) and xs[i][0] <= lo:
            active.append(xs[i])
            i += 1
        active = [x for x in active if x[1] > lo]
        if active:
            out[max(active, key=lambda x: x[0])[2]] += hi - lo
        else:
            out["idle"] += hi - lo
    return out


def main():
    a = sys.argv[1:]
    path = a[0]
    npass = int(a[a.index("--passes") + 1]) if "--passes" in a else 20
    mink = int(a[a.index("--min-kernels") + 1]) if "--min-kernels" in a else 1500
    ps = [p for p in passes(load(path)) if len(p) >= mink]
    ps = ps[-npass:]
    print(f"{len(ps)} big passes (>= {mink} kernels); kernels per pass median {st.median(len(p) for p in ps):.0f}")
    spans = [(max(x[0] + x[1] for x in p) - p[0][0]) / 1e6 for p in ps]
    print(f"span per pass: median {st.median(spans):.2f} ms (min {min(spans):.2f}, max {max(spans):.2f})")
    calls, sums, s0, crit = defaultdict(float), defaultdict(float), defaultdict(float), defaultdict(float)
    for p in ps:
        s0id = p[0][5]
        for a_, d, n, *rest in p:
            c = klass(n)
            calls[c] += 1
            sums[c] += d
            if rest[2] == s0id:
                s0[c] += d
        for c, v in partition(p, klass).items():
            crit[c] += v
    k = len(ps)
    print(f"{'class':16s} {'calls':>8s} {'sum ms':>8s} {'s0 ms':>8s} {'crit ms':>8s}")
    for c in sorted(crit, key=lambda c: -crit[c]):
        print(f"{c:16s} {calls[c]/k:8.1f} {sums[c]/k/1e6:8.3f} {s0[c]/k/1e6:8.3f} {crit[c]/k/1e6:8.3f}")
    dense = [c for c in crit if c.startswith("dense")]
    print(f"{'dense total':16s} {sum(calls[c] for c in dense)/k:8.1f} {sum(sums[c] for c in dense)/k/1e6:8.3f} "
          f"{sum(s0[c] for c in dense)/k/1e6:8.3f} {sum(crit[c] for c in dense)/k/1e6:8.3f}")
    print(f"{'all':16s} {sum(calls.values())/k:8.1f} {sum(sums.values())/k/1e6:8.3f} {sum(s0.values())/k/1e6:8.3f} {sum(crit.values())/k/1e6:8.3f}")
    # shapes
    shp = defaultdict(list)
    for p in ps:
        for a_, d, n, gx, gy, sm in p:
            c = klass(n)
            if c.startswith("dense") and c not in ("dense.cast",):
                key = (short(n) + ("<" + n.split("<")[1].split(">")[0] + ">" if "<" in n.split("(")[0] and c == "dense.main" else ""), gx, gy)
                shp[key].append(d)
    print("\ndense kernels by (kernel, grid x, grid y): calls per pass, median us, sum ms per pass")
    for key, v in sorted(shp.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {key[0]:38s} {key[1]:>6s} {key[2]:>4s} {len(v)/k:7.1f} {st.median(v)/1e3:8.1f} {sum(v)/k/1e6:8.3f}")


if __name__ == "__main__":
    main()
