#!/usr/bin/env python3
"""postchain2 step 3: per server, per graph launch of the main captured step (>= 1000 nodes, middle 10-70 % of MID1's launches):
span (first start .. last end), union (time at least one node runs), idle = span - union, busy (sum of node durations),
overlap = busy - union (time two branches run at once), side = busy of the side stream's nodes, side_hidden = the part of the side
stream's node time that overlaps a main-stream node. Medians in ms. usage: nsys_ovl.py X.sqlite [...]"""
import sqlite3, sys, collections, statistics as st
def union(iv):
    tot, cur_s, cur_e = 0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None: tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else: cur_e = max(cur_e, e)
    return tot + (cur_e - cur_s if cur_e is not None else 0)
def inter(a, b):
    # total length of (union a) intersect (union b); a, b lists of intervals
    ua = sorted(a); ub = sorted(b); i = j = 0; tot = 0
    def merge(x):
        out = []
        for s, e in x:
            if out and s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
            else: out.append([s, e])
        return out
    ua, ub = merge(ua), merge(ub)
    while i < len(ua) and j < len(ub):
        s, e = max(ua[i][0], ub[j][0]), min(ua[i][1], ub[j][1])
        if e > s: tot += e - s
        if ua[i][1] < ub[j][1]: i += 1
        else: j += 1
    return tot
print(f"{'server':10s} {'launches':>8s} {'span':>8s} {'union':>8s} {'idle':>7s} {'busy':>8s} {'overlap':>8s} {'main':>8s} {'side':>7s} {'side_hid':>8s}")
for f in sys.argv[1:]:
    c = sqlite3.connect(f)
    by = collections.defaultdict(list)
    for s, e, corr, stream in c.execute("select start, end, correlationId, streamId from CUPTI_ACTIVITY_KIND_KERNEL where graphNodeId is not null"):
        by[corr].append((s, e, stream))
    ls = sorted((v for v in by.values() if len(v) >= 1000), key=lambda k: min(x[0] for x in k))
    ls = ls[int(len(ls) * 0.10):int(len(ls) * 0.70)]
    main_stream = collections.Counter(x[2] for x in ls[0]).most_common(1)[0][0]
    rows = []
    for k in ls:
        iv = [(s, e) for s, e, _ in k]
        m = [(s, e) for s, e, sid in k if sid == main_stream]; sd = [(s, e) for s, e, sid in k if sid != main_stream]
        sp = max(e for _, e in iv) - min(s for s, _ in iv); un = union(iv); bu = sum(e - s for s, e in iv)
        rows.append((sp, un, sp - un, bu, bu - un, sum(e - s for s, e in m), sum(e - s for s, e in sd), inter(sd, m)))
    md = [st.median(x[i] for x in rows) / 1e6 for i in range(8)]
    print(f"{f.split('/')[-1][:-7]:10s} {len(rows):8d} " + " ".join(f"{v:8.3f}" for v in md))
