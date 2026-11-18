#!/usr/bin/env python3
"""postchain2 step 3: where the idle time of a graph launch is. Per launch (main step, middle 10-70 % of MID1), the idle gaps (no node
running) longer than 20 us, keyed by (the node ordinal, in capture order, that ends right before the gap, the one that starts after it);
per key: in how many launches it occurs and the median gap (us). usage: nsys_gap.py X.sqlite [...]"""
import sqlite3, sys, collections, statistics as st
for f in sys.argv[1:]:
    c = sqlite3.connect(f)
    S = dict(c.execute("select id, value from StringIds"))
    by = collections.defaultdict(list)
    for s, e, corr, n, node, sid in c.execute("select start, end, correlationId, shortName, graphNodeId, streamId from CUPTI_ACTIVITY_KIND_KERNEL where graphNodeId is not null"):
        by[corr].append((node, s, e, S[n], sid))
    for t in ("CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMSET"):
        for s, e, corr, node, sid in c.execute(f"select start, end, correlationId, graphNodeId, streamId from {t} where graphNodeId is not null"):
            by[corr].append((node, s, e, t.split("_")[-1].lower(), sid))
    ls = sorted((sorted(v) for v in by.values() if len(v) >= 1000), key=lambda k: min(x[1] for x in k))
    ls = ls[int(len(ls) * 0.10):int(len(ls) * 0.70)]
    gaps = collections.defaultdict(list)
    for k in ls:
        ordk = [(s, e, i, nm, sid) for i, (_, s, e, nm, sid) in enumerate(k)]
        ev = sorted(ordk)
        cur_e, last = None, None
        for s, e, i, nm, sid in ev:
            if cur_e is not None and s - cur_e > 20000:
                gaps[(last[2], last[3], i, nm)].append((s - cur_e) / 1e3)
            if cur_e is None or e > cur_e:
                cur_e, last = e, (s, e, i, nm, sid)
    print(f"== {f.split('/')[-1]}: {len(ls)} launches")
    for key, v in sorted(gaps.items(), key=lambda x: -sum(x[1]))[:6]:
        print(f"   after #{key[0]} {key[1][:26]:26s} before #{key[2]} {key[3][:26]:26s}: {len(v):4d} launches, median gap {st.median(v):7.1f} us")
