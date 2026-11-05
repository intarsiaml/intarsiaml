#!/usr/bin/env python3
"""Lane dsv41/decode4: per-layer segments of a verify pass anchored on the two grouped_cp kernels (ours and TensorFold have both).

Usage: layer_seg.py ours|tf TRACE.csv [ROWS]
Per verify pass of ROWS rows (default 3) and per layer: gu = first grouped_cp duration (gate/up), gap1 = its end to the second grouped_cp's start,
dn = second grouped_cp duration, rest = its end to the next layer's first grouped_cp start (combine, collectives, hyper-connection, attention,
router chain). Medians over layers and passes, in us, and the sum per pass in ms. Layer 0 and the last are dropped.
"""
import statistics as st
import sys

sys.path.insert(0, __import__("os").path.dirname(__file__))
from decode4_table import load, rounds_ours, rounds_tf, split_ours, split_tf, short  # noqa: E402

tf = sys.argv[1] == "tf"
rows = load(sys.argv[2])
want = int(sys.argv[3]) if len(sys.argv) > 3 else 3
rs = (rounds_tf(split_tf(rows)) if tf else rounds_ours(split_ours(rows)))[1:]
ps = [v for r, v, d in rs if r == want]
seg = {k: [] for k in ("gu", "gap1", "dn", "rest")}
tot = {k: [] for k in seg}
nl = []
for p in ps:
    g = sorted((a, a + d) for a, d, nm, *_ in p if short(nm) == "grouped_cp_kernel")
    pairs = list(zip(g[0::2], g[1::2]))
    nl.append(len(pairs))
    acc = {k: 0.0 for k in seg}
    for i, (x, y) in enumerate(pairs[1:-1], 1):
        v = {"gu": x[1] - x[0], "gap1": y[0] - x[1], "dn": y[1] - y[0], "rest": pairs[i + 1][0][0] - y[1]}
        for k in seg:
            seg[k].append(v[k] / 1e3)
            acc[k] += v[k] / 1e6
    for k in seg:
        tot[k].append(acc[k])
print(f"{sys.argv[1]} {want} rows: {len(ps)} passes, layers {st.median(nl)}")
for k in seg:
    print(f"  {k:5s} median {st.median(seg[k]):7.1f} us   mean {st.mean(seg[k]):7.1f}   per pass (38 layers) {st.median(tot[k]):6.2f} ms")

# ---- phases of "rest": anchors common to both engines
import re  # noqa: E402

FRONT = re.compile(r"^(rot_in_kernel|rot_many_kernel|split_kernel|linear_kernel|glinear_kernel)$")
ATT = re.compile(r"^(attn_sparse_mma|_sparse_attn_part)$")
MERGE = re.compile(r"^(attn_sparse_merge|_sparse_attn_merge_rot2)$")
STAGE = re.compile(r"rdma_stage$")
GATE = re.compile(r"^(dense_f16w_vec_kernel|_rowmm_parts)$")
ph = {k: [] for k in ("a>b coll+hc", "b>c front proj", "c>d attention", "d>e out proj", "e>f coll+hc", "f>g router chain")}
for p in ps:
    ks = sorted((a, a + d, short(nm)) for a, d, nm, *_ in p)
    g = sorted((a, a + d) for a, d, nm, *_ in p if short(nm) == "grouped_cp_kernel")
    pairs = list(zip(g[0::2], g[1::2]))
    for i, (x, y) in enumerate(pairs[1:-1], 1):
        a_, g_ = y[1], pairs[i + 1][0][0]
        seq = [k for k in ks if a_ <= k[0] < g_]
        try:
            b = next(k[0] for k in seq if FRONT.match(k[2]))
            c = next(k[0] for k in seq if ATT.match(k[2]) and k[0] >= b)
            d = next(k[1] for k in seq if MERGE.match(k[2]) and k[0] >= c)
            e = next(k[0] for k in seq if STAGE.search(k[2]) and k[0] >= d)
            f = next(k[0] for k in seq if GATE.match(k[2]) and k[0] >= e)
        except StopIteration:
            continue
        for key, (u, v) in zip(ph, ((a_, b), (b, c), (c, d), (d, e), (e, f), (f, g_))):
            ph[key].append((v - u) / 1e3)
print("  phases of rest (us, median / mean):", {k: (round(st.median(v), 1), round(st.mean(v), 1)) for k, v in ph.items() if v}, "layers", len(next(iter(ph.values()))))
