#!/usr/bin/env python3
"""sideeager: one nsys sqlite of a whole serve session (nsys2.sh: collection starts when /health answers). Graph launches (>= 900 kernel nodes: the 1002 / 1000 / 994-node variants of with a graph node id, the
captured MTP step) are classified slow / fast by the idle time inside the launch (span minus the union of node intervals) > THR us (nsys_mode.py's rule), ordered in time and cut
into requests where two consecutive launches are more than GAP ms apart. Per request: launches, slow launches, the runs of slow launches (first-last index within the request),
and the eager (non-graph) kernels per stream in the gap before the request and inside it ("S" = a stream other than the most common eager stream, i.e. the side stream).
The first slow launch of the session and what preceded it are printed. usage: nsys_req.py X.sqlite [THR=200] [GAP=60]"""
import sqlite3, sys, collections, statistics as st
f = sys.argv[1]; THR = float(sys.argv[2]) if len(sys.argv) > 2 else 200; GAP = float(sys.argv[3]) if len(sys.argv) > 3 else 60
c = sqlite3.connect(f)
by = collections.defaultdict(list)
for s, e, corr in c.execute("select start, end, correlationId from CUPTI_ACTIVITY_KIND_KERNEL where graphNodeId is not null"):
    by[corr].append((s, e))
ls = sorted((v for v in by.values() if len(v) >= 900), key=lambda k: min(x[0] for x in k))
def idle(k):
    cur = None; tot = 0
    for s, e in sorted(k):
        if cur is not None and s > cur: tot += s - cur
        cur = e if cur is None else max(cur, e)
    return tot / 1e3
eag = sorted(c.execute("select start, end, streamId from CUPTI_ACTIVITY_KIND_KERNEL where graphNodeId is null"))
main = collections.Counter(x[2] for x in eag).most_common(1)[0][0] if eag else None
t0 = [min(x[0] for x in k) for k in ls]; t1 = [max(x[1] for x in k) for k in ls]; idl = [idle(k) for k in ls]; slow = [x > THR for x in idl]
segs = []; cur = [0]
for i in range(1, len(ls)):
    if (t0[i] - t1[i - 1]) / 1e6 > GAP: segs.append(cur); cur = []
    cur.append(i)
segs.append(cur)
def runs(idx):
    out = []; a = None
    for j, i in enumerate(idx + [None]):
        s = i is not None and slow[i]
        if s and a is None: a = j
        if not s and a is not None: out.append(f"{a}-{j - 1}"); a = None
    return out
def eager(lo, hi):
    cnt = collections.Counter("M" if x[2] == main else "S" for x in eag if lo <= x[0] < hi)
    return f"M {cnt['M']} S {cnt['S']}"
print(f"{f}: {len(ls)} graph launches, slow {sum(slow)}, {len(segs)} requests (gap > {GAP} ms), main eager stream {main}")
prev_end = 0
for n, idx in enumerate(segs):
    lo, hi = t0[idx[0]], t1[idx[-1]]
    gaps = [(t0[i] - t1[i - 1]) / 1e3 for i in idx[1:]]
    print(f" req {n}: {len(idx):4d} launches, slow {sum(slow[i] for i in idx):4d}, runs {' '.join(runs(idx)) or '-'}; median idle {st.median(idl[i] for i in idx):.1f} us, span {st.median((t1[i] - t0[i]) / 1e6 for i in idx):.3f} ms, launch gap {st.median(gaps) if gaps else 0:.1f} us (launch-to-launch {st.median((t0[i] - t0[i - 1]) / 1e6 for i in idx[1:]) if gaps else 0:.2f} ms); eager before [{eager(prev_end, lo)}] inside [{eager(lo, hi)}]")
    prev_end = hi
first = next((i for i, s in enumerate(slow) if s), None)
if first is None: print("no slow launch in the session")
else:
    n = next(n for n, idx in enumerate(segs) if first in idx); k = segs[n].index(first)
    pe = t1[segs[n - 1][-1]] if n else 0
    print(f"first slow launch: request {n}, launch {k} of {len(segs[n])}; eager kernels since the session start [{eager(0, t0[first])}], since the previous request's end [{eager(pe, t0[first])}]")
    side = [x for x in eag if x[2] != main and x[0] < t0[first]]
    print(f"side-stream eager kernels before it: {len(side)}" + (f", last at {(t0[first] - side[-1][1]) / 1e6:.1f} ms before it" if side else ""))
