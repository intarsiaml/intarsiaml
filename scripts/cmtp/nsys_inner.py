#!/usr/bin/env python3
"""Lane cmtp W1: GPU busy/idle INSIDE the captured spec-step graph from a node-level nsys export (--cuda-graph-trace=node).
  nsys_inner.py <file.sqlite>   passes = runs of graph kernels split where the gap to the previous kernel exceeds 40 us."""
import sqlite3, sys, statistics as st, collections
db = sqlite3.connect(sys.argv[1])
ks = db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL where graphNodeId is not null order by start").fetchall()
names = dict(db.execute("select id, value from StringIds"))
nm = lambda x: names.get(x, str(x))
passes, cur = [], [ks[0]]
for k in ks[1:]:
    if k[0] - max(c[1] for c in cur[-8:]) > 40e3: passes.append(cur); cur = [k]
    else: cur.append(k)
passes.append(cur)
passes = [p for p in passes if len(p) > 200]
print(f"graph kernels {len(ks)}, passes {len(passes)}, kernels/pass med {st.median(len(p) for p in passes):.0f}")
span, busy, gaps = [], [], []
big = collections.Counter(); bigt = collections.Counter(); sizes = []
for p in passes:
    t0, t1 = p[0][0], max(k[1] for k in p)
    cov = 0; cur_end = t0
    for k in p:
        s = max(k[0], cur_end)
        if k[1] > s: cov += k[1] - s; cur_end = k[1]
    span.append(t1 - t0); busy.append(cov)
    prev_end = p[0][1]
    for k in p[1:]:
        g = k[0] - prev_end
        if g > 0:
            sizes.append(g)
            if g > 5e3: big[(nm(p[p.index(k) - 1][2]) if False else '', nm(k[2]))] += 1; bigt[nm(k[2])] += g
        prev_end = max(prev_end, k[1])
sp, bu = sum(span), sum(busy)
print(f"pass span mean {sp/len(span)/1e6:.3f} ms; kernel-busy {bu/len(span)/1e6:.3f} ms; inside-graph idle {(sp-bu)/len(span)/1e6:.3f} ms = {100*(sp-bu)/sp:.2f} % of the graph span")
print(f"inter-kernel gaps: n {len(sizes)}, median {st.median(sizes)/1e3:.2f} us, p90 {sorted(sizes)[9*len(sizes)//10]/1e3:.2f} us, sum/pass {sum(sizes)/len(passes)/1e3:.1f} us")
print("kernels most often after a >5 us gap (name: gaps/pass, us/pass):")
for n, t in bigt.most_common(8): print(f"  {n[:60]:60s} {sum(c for (a, b), c in big.items() if b == n)/len(passes):6.2f} {t/len(passes)/1e3:8.1f}")
