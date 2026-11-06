#!/usr/bin/env python3
"""Lane dsv41/decode4: one class table per pass kind for our decode and TensorFold's, from nsys cuda_gpu_trace CSVs.

Usage: decode4_table.py ours TRACE.csv     (ours: passes split at the logits all-gather)
       decode4_table.py tf   TRACE.csv     (TensorFold: passes start at `_embed_init`, rows = its grid x)
Both are classed by kernel name into the same classes. Per pass kind the table gives, in ms per pass (median over passes):
  ksum  sum of kernel durations (a kernel that sits in griddepcontrol.wait counts its waiting, so overlap makes it exceed the span)
  crit  critical-path share: every instant of the pass is charged to the class of the running kernel that started last; the
        L2-prefetch side stream is left out (it is hidden); instants with no kernel running are `idle`. The crit column sums to the span.
Then the round: draft-pass start to the next draft-pass start (the period).
"""
import csv
import re
import statistics as st
import sys
from collections import defaultdict

CLASSES = ["dense", "moe", "coll", "hc", "attn", "other", "prefetch"]


def short(n):
    n = n.replace("void ", "").replace("(anonymous namespace)::", "").replace("<unnamed>::", "")
    n = re.split(r"[<(]", n)[0].split("::")[-1]
    return n[:60]


def cls(n, tf):
    s = short(n).lower()
    if "prefetch" in s:
        return "prefetch"
    if re.search(r"^nccl|rdma|f32_to_bf16|tf_ds_add2|^ik_ring|allreduce", s):
        return "coll"
    if tf:
        if re.search(r"grouped_|down_combine|gateup", s):
            return "moe"
        if re.search(r"^_gemm|glinear|unpack|kernel2|rot_in|rot_many|rot_out|rowmm|_rmsnorm_rot", s):
            return "dense"
        if re.search(r"_hc_|hc_", s):
            return "hc"
        if re.search(r"sparse_attn|index_score|topk|rope|kv_|_q_kv|compress|swa|attn|_finish$", s):
            return "attn"
        return "other"
    if re.search(r"grouped_cp|group_kernel|epilogue|combine|silu_glu", s):
        return "moe"
    if re.search(r"linear_kernel|split_kernel|split_finish|dense_f16|rot_in|rot_out|rot_many|qdq|transpose01|add_f32|cast", s):
        return "dense"
    if re.search(r"^hc_", s):
        return "hc"
    if re.search(r"attn_|rope|index_|window_build|ring_write|kv_|compress|sel_|topk", s):
        return "attn"
    return "other"


def load(path):
    rows = []
    with open(path, newline="") as f:
        lines = [l for l in f if l.strip()]
    i0 = next(i for i, l in enumerate(lines) if l.startswith("Start") or l.startswith('"Start'))
    for d in csv.DictReader(lines[i0:]):
        n = d["Name"]
        if n.startswith("[CUDA"):
            continue
        rows.append((int(d["Start (ns)"]), int(d["Duration (ns)"]), n, d["GrdX"], d["Strm"]))
    rows.sort()
    return rows


def partition(p, tf):
    xs = sorted((a, a + d, cls(n, tf)) for a, d, n, _, _ in p)
    xs = [x for x in xs if x[2] != "prefetch"]
    pts = sorted({t for a, b, _ in xs for t in (a, b)})
    out = defaultdict(float)
    active, i = [], 0
    for a_, b_ in zip(pts, pts[1:]):
        while i < len(xs) and xs[i][0] <= a_:
            active.append(xs[i])
            i += 1
        active = [x for x in active if x[1] > a_]
        if active:
            out[max(active, key=lambda x: x[0])[2]] += b_ - a_
        else:
            out["idle"] += b_ - a_
    return out


def split_tf(rows):
    idx = [i for i, r in enumerate(rows) if r[2].startswith("_embed_init")]
    return [(int(rows[a][3]), rows[a:b]) for a, b in zip(idx, idx[1:] + [len(rows)])]


def split_ours(rows):
    out, cur = [], []
    for r in rows:
        cur.append(r)
        if "AllGather" in r[2]:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def table(name, ps, tf):
    if not ps:
        return
    n = len(ps)
    spans = [(max(a + d for a, d, *_ in p) - p[0][0]) / 1e6 for p in ps]
    ks, cr, cnt = defaultdict(list), defaultdict(list), defaultdict(list)
    for p in ps:
        k = defaultdict(float)
        c = defaultdict(int)
        for a, d, nm, _, _ in p:
            k[cls(nm, tf)] += d / 1e6
            c[cls(nm, tf)] += 1
        part = partition(p, tf)
        for cl in CLASSES + ["idle"]:
            ks[cl].append(k.get(cl, 0.0))
            cr[cl].append(part.get(cl, 0.0) / 1e6)
            cnt[cl].append(c.get(cl, 0))
    print(f"\n== {name}: {n} passes, span median {st.median(spans):.2f} ms, kernels {st.median(len(p) for p in ps):.0f}")
    print(f"  {'class':9s} {'ksum':>8s} {'crit':>8s} {'launches':>9s}")
    for cl in CLASSES + ["idle"]:
        print(f"  {cl:9s} {st.median(ks[cl]):8.2f} {st.median(cr[cl]):8.2f} {st.median(cnt[cl]):9.0f}")
    print(f"  {'total':9s} {sum(st.median(ks[c]) for c in CLASSES):8.2f} {sum(st.median(cr[c]) for c in CLASSES + ['idle']):8.2f}")


def rounds_ours(passes):
    """(rows, verify pass, draft pass) per round: a big pass, then the small chunk whose head holds the verify's argmax (grid x = rows)."""
    out = []
    for a, b in zip(passes, passes[1:]):
        if len(a) > 700 and 20 <= len(b) <= 700:
            am = [x for x in b[:6] if "argmax" in x[2]]
            if am:
                out.append((int(am[0][3]), a, b))
    return out


def rounds_tf(passes):
    out = []
    for (t, p), (t2, q) in zip(passes, passes[1:]):
        if 2 <= t <= 5 and t != 5 and t2 == 5:
            out.append((t, p, q))
    return out


def end(p):
    return max(a + d for a, d, *_ in p)


def round_stats(name, rs):
    """Median period of the rounds (rs: rows, verify, draft in time order) and the idle gaps over 50 us inside a round: with the
    graphs the chunks split at the all-gather are not the host's boundaries, so the gaps are found on the merged timeline."""
    if len(rs) < 4:
        return
    per = [(y[1][0][0] - x[1][0][0]) / 1e6 for x, y in zip(rs, rs[1:])]
    print(f"\n== {name} rounds: {len(rs)}; period {st.median(per):.2f} ms (mean {st.mean(per):.2f})")
    big, tot, sizes = [], [], []
    for x, y in zip(rs, rs[1:]):
        ev = sorted((a, a + d) for p in (x[1], x[2]) for a, d, *_ in p)
        gaps, cur = [], ev[0][1]
        for a, b in ev[1:]:
            if a - cur > 50_000:
                gaps.append((a - cur) / 1e6)
            cur = max(cur, b)
        tot.append(sum(gaps))
        sizes.append(sorted(gaps, reverse=True)[:2])
    print(f"   idle gaps over 50 us per round (verify chunk + draft chunk): total median {st.median(tot):.3f} ms; largest two, median {st.median(g[0] for g in sizes if g):.3f} and {st.median(g[1] for g in sizes if len(g) > 1):.3f} ms")
    for r in sorted({r for r, *_ in rs}):
        sub = [(x, y) for x, y in zip(rs, rs[1:]) if x[0] == r]
        print(f"   rows {r}: {sum(1 for x in rs if x[0] == r)} rounds, period median {st.median((y[1][0][0] - x[1][0][0]) / 1e6 for x, y in sub):.2f}")


def main():
    a = sys.argv[1:]
    tf = a[0] == "tf"
    rows = load(a[1])
    if tf:
        passes = split_tf(rows)
        rs = rounds_tf(passes)
        # ordered by time; drop the first round (warm-up)
        rs = rs[1:]
        for r in sorted({r for r, *_ in rs}):
            table(f"TF verify {r} rows", [v for rr, v, d in rs if rr == r], True)
        table("TF draft block", [d for _, v, d in rs], True)
        round_stats("TF", rs)
    else:
        passes = split_ours(rows)
        rs = rounds_ours(passes)[1:]
        for r in sorted({r for r, *_ in rs}):
            ps = [v for rr, v, d in rs if rr == r]
            if len(ps) >= 3:
                table(f"ours verify {r} rows", ps, False)
        table("ours draft block", [d for _, v, d in rs], False)
        round_stats("ours", rs)


if __name__ == "__main__":
    main()
