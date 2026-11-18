#!/usr/bin/env python3
"""sideeager step 3: per-kernel device times of the graph replays in one nsys sqlite (MID1 of one server).
Kernels / memcpys / memsets with a graph node id (ordered by node id = capture order) are grouped by the API call that launched them (correlationId of cudaGraphLaunch);
launches are grouped by their kernel-name sequence (one per captured graph variant). Per variant, the launches of the middle of the request
(ordinal fraction [LO, HI), default 0.10 .. 0.70: past the first steps, before n20's own MID1 chain at 665) give per launch:
span (first kernel start .. last kernel end), busy (sum of node durations), gaps (span - busy); and per node ordinal its duration.
Writes JSON {variant: {"names": [...], "n": launches, "span": [...], "busy": [...], "dur": [[per ordinal durations]...]}}.
usage: nsys_an.py X.sqlite OUT.json [LO HI]"""
import sqlite3, sys, json, collections, hashlib
db, out = sys.argv[1], sys.argv[2]
LO, HI = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 else (0.10, 0.70)
c = sqlite3.connect(db)
tabs = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
S = dict(c.execute("select id, value from StringIds"))
rows = []
for r in c.execute("select start, end, correlationId, shortName, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL where graphNodeId is not null"):
    rows.append((r[0], r[1], r[2], S.get(r[3], str(r[3])), r[4]))
for t, nm in (("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "memset")):
    if t in tabs:
        cols = {x[1] for x in c.execute(f"pragma table_info({t})")}
        if "graphNodeId" in cols:
            for r in c.execute(f"select start, end, correlationId, graphNodeId from {t} where graphNodeId is not null"):
                rows.append((r[0], r[1], r[2], nm, r[3]))
by = collections.defaultdict(list)
for s, e, corr, nm, node in rows:
    by[corr].append((node, s, e, nm))
var = collections.defaultdict(list)
for corr, ks in by.items():
    # node order = capture order (graph node ids), stable across launches even where branches run concurrently
    ks.sort()
    ks = [(s, e, nm) for _, s, e, nm in ks]
    var[tuple(k[2] for k in ks)].append(ks)
res = {}
for names, ls in var.items():
    ls.sort(key=lambda k: min(x[0] for x in k))
    n = len(ls)
    if n < 20:
        continue
    sel = ls[int(n * LO):int(n * HI)]
    key = f"v{len(names)}_" + hashlib.sha1("|".join(names).encode()).hexdigest()[:8]
    res[key] = {"names": list(names), "n_all": n, "n": len(sel),
                "span": [(max(x[1] for x in k) - min(x[0] for x in k)) / 1e6 for k in sel],
                "busy": [sum(e - s for s, e, _ in k) / 1e6 for k in sel],
                "t0": [min(x[0] for x in k) for k in sel],
                "dur": [[(k[i][1] - k[i][0]) / 1e3 for k in sel] for i in range(len(names))]}
json.dump(res, open(out, "w"))
for k, v in sorted(res.items(), key=lambda x: -x[1]["n_all"]):
    sp = sorted(v["span"]); bu = sorted(v["busy"])
    print(f"{k}: {len(v['names'])} nodes, {v['n_all']} launches ({v['n']} used), median span {sp[len(sp)//2]:.3f} ms, busy {bu[len(bu)//2]:.3f} ms")
