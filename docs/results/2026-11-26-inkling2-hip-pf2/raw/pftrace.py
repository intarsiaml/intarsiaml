#!/usr/bin/env python3
"""Prefill split of a rocprofv3 kernel trace (csv) of `intarsia generate` (lane ink-hip-pf2).
Chunks are the segments that start at an embedding dispatch; the first `nchunks` segments are the prefill chunks (the rest is
decode). Per chunk: wall (first start .. last end), kernel busy, idle inside the chunk, the gap before the chunk (previous end ..
its embedding start), and per kernel class / kernel: calls, ms. usage: pftrace.py trace.csv nchunks [top=40]"""
import collections, csv, re, sys

rows = sorted(csv.DictReader(open(sys.argv[1])), key=lambda r: int(r["Start_Timestamp"]))
nch = int(sys.argv[2])
top = int(sys.argv[3]) if len(sys.argv) > 3 else 40


def short(n):
    n = n.replace("(anonymous namespace)::", "")
    n = re.sub(r"\(.*$", "", n)
    n = re.sub(r"^void ", "", n)
    return n.replace(" [clone .kd]", "")[:90]


CLASSES = [  # (class, regex on the short name), first match wins
    ("experts", r"ink_moe|moe_wmma"),
    ("router-topk", r"moe_route"),
    ("attention", r"rel_sink|ink_attn|flash|ink_prologue|ink_ring|ink_kv"),
    ("mix-row", r"ink_row|ink_ring_kernel|ink_blend|ink_ctl|ink_sconv|ink_conv|ink_mix|ink_res"),
    ("gemm", r"matmul|gemm|wmma|logits|gemv"),
    ("norm-embed", r"rmsnorm|embedding|norm"),
    ("argmax", r"argmax"),
]


def klass(n):
    for c, rx in CLASSES:
        if re.search(rx, n):
            return c
    return "other"


emb = [i for i, r in enumerate(rows) if re.search(r"embedding", r["Kernel_Name"])]
if len(emb) < nch:
    sys.exit(f"only {len(emb)} embedding dispatches")
bounds = emb[: nch + 1] if len(emb) > nch else emb[:nch] + [len(rows)]
tot = collections.defaultdict(lambda: [0, 0.0])
tcls = collections.defaultdict(float)
for c in range(nch):
    lo, hi = bounds[c], bounds[c + 1]
    seg = rows[lo:hi]
    # the last prefill chunk ends with the head: cut the segment at its argmax (the decode steps follow)
    st0 = int(seg[0]["Start_Timestamp"])
    en = max(int(r["End_Timestamp"]) for r in seg)
    busy = sum(int(r["End_Timestamp"]) - int(r["Start_Timestamp"]) for r in seg) / 1e6
    wall = (en - st0) / 1e6
    gap = (st0 - int(rows[lo - 1]["End_Timestamp"])) / 1e6 if lo > 0 else 0.0
    grid = seg[0]["Grid_Size_X"]
    big_gaps = []
    for i in range(1, len(seg)):
        g = (int(seg[i]["Start_Timestamp"]) - int(seg[i - 1]["End_Timestamp"])) / 1e3
        if g > 50:
            big_gaps.append((g, short(seg[i]["Kernel_Name"])))
    print(f"chunk {c}: {len(seg)} dispatches, wall {wall:.1f} ms, busy {busy:.1f} ms, idle {wall - busy:.1f} ms "
          f"({100 * (wall - busy) / wall:.1f} %), gap before {gap:.2f} ms, emb grid {grid}; idle gaps > 50 us: {len(big_gaps)} "
          f"({sum(g for g, _ in big_gaps) / 1e3:.1f} ms)")
    g = collections.defaultdict(lambda: [0, 0.0, set()])
    for r in seg:
        k = short(r["Kernel_Name"])
        d = (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e6
        g[k][0] += 1
        g[k][1] += d
        g[k][2].add((r["VGPR_Count"], r["LDS_Block_Size"], r["Workgroup_Size_X"]))
        tot[k][0] += 1
        tot[k][1] += d
        tcls[klass(k)] += d
    cl = collections.defaultdict(float)
    for k, (n, t, _) in g.items():
        cl[klass(k)] += t
    print("   classes ms: " + ", ".join(f"{c} {t:.1f}" for c, t in sorted(cl.items(), key=lambda kv: -kv[1])))
    if c == 0 or c == nch - 2 or "-v" in sys.argv:
        print(f"   {'kernel':90s} {'n':>4s} {'ms':>8s} {'us/call':>9s}  vgpr/lds/wg")
        for k, (n, t, meta) in sorted(g.items(), key=lambda kv: -kv[1][1])[:top]:
            m = ";".join("/".join(x) for x in sorted(meta))[:40]
            print(f"   {k:90s} {n:4d} {t:8.2f} {1e3 * t / n:9.1f}  {m}")
    for gg, k in sorted(big_gaps, reverse=True)[:5]:
        print(f"     gap {gg:8.1f} us before {k}")
print("all prefill chunks, classes ms: " + ", ".join(f"{c} {t:.1f}" for c, t in sorted(tcls.items(), key=lambda kv: -kv[1])))
