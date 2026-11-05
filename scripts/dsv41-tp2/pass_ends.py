#!/usr/bin/env python3
"""Lane dsv41/decode4: prologue (pass start to first grouped_cp) and epilogue (last grouped_cp end to pass end) of a verify pass, ours and TF.
Usage: pass_ends.py ours|tf TRACE.csv ROWS [--list]"""
import statistics as st
import sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from decode4_table import load, rounds_ours, rounds_tf, split_ours, split_tf, short, cls, end  # noqa: E402

tf = sys.argv[1] == "tf"
rows = load(sys.argv[2])
want = int(sys.argv[3])
rs = (rounds_tf(split_tf(rows)) if tf else rounds_ours(split_ours(rows)))[1:]
ps = [v for r, v, d in rs if r == want]
pro, epi, span, mid = [], [], [], []
for p in ps:
    g = sorted((a, a + d) for a, d, nm, *_ in p if short(nm) == "grouped_cp_kernel")
    t0 = min(a for a, *_ in p)
    t1 = end(p)
    pro.append((g[0][0] - t0) / 1e6)
    epi.append((t1 - g[-1][1]) / 1e6)
    span.append((t1 - t0) / 1e6)
    mid.append((g[-1][1] - g[0][0]) / 1e6)
print(f"{sys.argv[1]} {want} rows, {len(ps)} passes: span {st.median(span):.2f}  prologue {st.median(pro):.2f}  first gu to last down end {st.median(mid):.2f}  epilogue {st.median(epi):.2f}")
if "--list" in sys.argv:
    p = ps[len(ps) // 2]
    g = sorted((a, a + d) for a, d, nm, *_ in p if short(nm) == "grouped_cp_kernel")
    t0 = min(a for a, *_ in p)
    print("-- prologue")
    for a, d, nm, gx, st_ in sorted(p):
        if a < g[0][0]:
            print(f"{(a - t0) / 1e3:9.1f} {d / 1e3:8.1f} s{st_:>3s} {short(nm)} g={gx}")
    print("-- epilogue")
    for a, d, nm, gx, st_ in sorted(p):
        if a >= g[-1][1] - 1:
            print(f"{(a - t0) / 1e3:9.1f} {d / 1e3:8.1f} s{st_:>3s} {short(nm)} g={gx}")
