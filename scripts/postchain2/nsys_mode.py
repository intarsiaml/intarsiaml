#!/usr/bin/env python3
"""postchain2 step 3: inside one server's MID1 trace, split the main step's launches by mode (idle inside the launch > THR us = slow)
and compare, per kernel name, the median node duration in slow vs fast launches (us, summed over the name's nodes per launch), plus
the median start-to-start latency between consecutive dependent main-stream nodes. usage: nsys_mode.py X.sqlite [THR=200]"""
import sqlite3, sys, collections, statistics as st
f = sys.argv[1]; THR = float(sys.argv[2]) if len(sys.argv) > 2 else 200
c = sqlite3.connect(f); S = dict(c.execute("select id, value from StringIds"))
by = collections.defaultdict(list)
for s, e, corr, n, node, sid in c.execute("select start, end, correlationId, shortName, graphNodeId, streamId from CUPTI_ACTIVITY_KIND_KERNEL where graphNodeId is not null"):
    by[corr].append((node, s, e, S[n], sid))
ls = sorted((sorted(v) for v in by.values() if len(v) >= 1000), key=lambda k: min(x[1] for x in k))
names = [x[3] for x in ls[0]]; ms = collections.Counter(x[4] for x in ls[0]).most_common(1)[0][0]
def idle(k):
    ev = sorted((s, e) for _, s, e, _, _ in k); cur = None; tot = 0
    for s, e in ev:
        if cur is not None and s > cur: tot += s - cur
        cur = e if cur is None else max(cur, e)
    return tot / 1e3
mode = [idle(k) > THR for k in ls]
eag = collections.Counter(r[0] for r in c.execute("select streamId from CUPTI_ACTIVITY_KIND_KERNEL where graphNodeId is null"))
print(f"{f}: {len(ls)} launches, slow {sum(mode)}; eager (non-graph) kernels per stream {dict(eag)}; sequence " + "".join("1" if m else "0" for m in mode))
dur = {True: collections.defaultdict(list), False: collections.defaultdict(list)}
lat = {True: [], False: []}
for k, m in zip(ls, mode):
    if len({x[3] for x in k}) and [x[3] for x in k] != names: continue
    per = collections.Counter()
    for _, s, e, nm, sid in k: per[nm] += (e - s) / 1e3
    for nm, v in per.items(): dur[m][nm].append(v)
    main = sorted((s, e) for _, s, e, _, sid in k if sid == ms)
    lat[m].append(st.median(b[0] - a[1] for a, b in zip(main, main[1:])) / 1e3)
if all(lat.values()):
    print(f"median gap end -> next start, main stream: slow {st.median(lat[True]):.3f} us, fast {st.median(lat[False]):.3f} us")
    rows = sorted(((st.median(dur[True][n]) - st.median(dur[False][n]), n, st.median(dur[False][n])) for n in dur[False] if dur[True][n]), reverse=True)
    for d, n, b in rows[:8] + rows[-5:]:
        print(f"   {n[:40]:40s} fast {b:8.1f} us  slow - fast {d:+7.1f} us")
