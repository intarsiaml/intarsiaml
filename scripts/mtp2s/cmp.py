#!/usr/bin/env python3
"""Lane mtp2s (2026-11-16): which kernels grow from a 3-row verify pass to a 6-row one (nsys sqlite of a node-level trace).

  cmp.py <file.sqlite> [kernels_a kernels_b]

Segments are cut at every device-to-host copy (as seg.py). Eager verify passes are told apart by kernel count: with the 35B's 40
layers (30 GDN + 10 full attention) the pass with the most segments of >= 1000 kernels per row-count class is taken; by default the two
most frequent eager segments of 800+ kernels, the smaller one being the 1-sequence (3-row) pass and the larger the 2-sequence (6-row) pass.
Prints per kernel name: launches and ms per pass in each, the difference, sorted by the difference; then per class totals and the
draft-pass (small eager segments) table.
"""
import collections, re, sqlite3, statistics as st, sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
from seg import cls  # noqa: E402


def load(path):
    db = sqlite3.connect(path)
    names = dict(db.execute("select id, value from StringIds"))
    ks = db.execute("select start, end, shortName, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    ks = [(s, e, names.get(sn, str(sn)), g is not None) for s, e, sn, g in ks]
    cps = db.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 2 order by start").fetchall()
    segs, ki = [], 0
    for cs, ce in cps:
        cur = []
        while ki < len(ks) and ks[ki][0] < cs:
            cur.append(ks[ki]); ki += 1
        if cur:
            segs.append(dict(ks=cur, n=len(cur), graph=sum(k[3] for k in cur) > len(cur) / 2, span=ce - cur[0][0]))
    return segs


def table(g):
    acc, cnt = collections.Counter(), collections.Counter()
    for s in g:
        for a, b, nm, _ in s["ks"]:
            acc[nm] += b - a; cnt[nm] += 1
    n = len(g)
    return {nm: (acc[nm] / n / 1e6, cnt[nm] / n, acc[nm] / cnt[nm]) for nm in acc}  # ms per pass, launches per pass, ns per launch


def main():
    segs = load(sys.argv[1])
    eager = collections.Counter(s["n"] for s in segs if not s["graph"] and s["n"] >= 800)
    if len(sys.argv) > 3:
        ka, kb = int(sys.argv[2]), int(sys.argv[3])
    else:
        ka, kb = sorted([k for k, _ in eager.most_common(2)])
    A = [s for s in segs if not s["graph"] and s["n"] == ka]; B = [s for s in segs if not s["graph"] and s["n"] == kb]
    print(f"eager passes by kernel count: {dict(eager.most_common(6))}")
    print(f"A = {ka} kernels (x{len(A)}, span {st.mean(s['span'] for s in A)/1e6:.2f} ms), B = {kb} kernels (x{len(B)}, span {st.mean(s['span'] for s in B)/1e6:.2f} ms)")
    ta, tb = table(A), table(B)
    rows = []
    for nm in set(ta) | set(tb):
        a, b = ta.get(nm, (0, 0, 0)), tb.get(nm, (0, 0, 0))
        rows.append((b[0] - a[0], nm, a, b))
    rows.sort(reverse=True)
    print("\nper kernel name: ms per pass A / B (launches per pass), difference B - A; top 30")
    for d, nm, a, b in rows[:30]:
        print(f"  {d:+7.3f}  A {a[0]:7.3f} ({a[1]:5.1f})  B {b[0]:7.3f} ({b[1]:5.1f})  {cls(nm, max(a[2], b[2])):16s} {nm[:90]}")
    ca, cb = collections.Counter(), collections.Counter()
    for nm, (t, _, per) in ta.items():
        ca[cls(nm, per)] += t
    for nm, (t, _, per) in tb.items():
        cb[cls(nm, per)] += t
    print("\nper class (kernel sum, ms per pass): A / B / B - A")
    for c in sorted(set(ca) | set(cb), key=lambda c: -(cb[c] - ca[c])):
        print(f"  {c:18s} {ca[c]:7.3f} {cb[c]:7.3f} {cb[c]-ca[c]:+7.3f}")
    print(f"  {'kernel sum':18s} {sum(ca.values()):7.3f} {sum(cb.values()):7.3f} {sum(cb.values())-sum(ca.values()):+7.3f}")
    small = collections.Counter(s["n"] for s in segs if not s["graph"] and s["n"] < 800)
    print(f"\nsmall eager segments (draft passes etc.) by kernel count: {dict(small.most_common(8))}")
    for k, c in small.most_common(4):
        g = [s for s in segs if not s["graph"] and s["n"] == k]
        t = table(g)
        print(f"\n  {k} kernels (x{c}, span {st.mean(s['span'] for s in g)/1e6:.3f} ms):")
        for nm, (ms, n, per) in sorted(t.items(), key=lambda x: -x[1][0])[:8]:
            print(f"    {ms:7.3f} ({n:4.1f})  {cls(nm, per):16s} {nm[:90]}")


if __name__ == "__main__":
    main()
