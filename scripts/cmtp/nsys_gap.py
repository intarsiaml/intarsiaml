#!/usr/bin/env python3
"""Lane cmtp W1: per-pass GPU idle from an nsys sqlite export (graph trace = each graph replay is one interval).
  nsys_gap.py <file.sqlite> [min_graph_ms]   (the 'pass' is the longest graph replays; shorter replays/kernels outside graphs are listed)
Busy = union of graph replays + eager kernels + memcpy + memset; idle = the rest of the span between pass starts. Reports per pass:
period, graph busy, idle (and by what sits in the gap: eager kernels, memcpy, runtime API calls with their durations)."""
import sqlite3, sys, statistics as st, collections
db = sqlite3.connect(sys.argv[1]); thr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
names = dict(db.execute("select id, value from StringIds"))
g = db.execute("select start, end, graphId, streamId from CUPTI_ACTIVITY_KIND_GRAPH_TRACE order by start").fetchall()
ks = db.execute("select start, end, shortName, graphNodeId from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
ks_out = [k for k in ks if not any(a[0] <= k[0] and k[1] <= a[1] for a in [])]  # filtered below
mc = db.execute("select start, end, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start").fetchall()
ms = db.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMSET order by start").fetchall()
rt = db.execute("select start, end, nameId from CUPTI_ACTIVITY_KIND_RUNTIME order by start").fetchall()
sy = db.execute("select start, end, syncType from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION order by start").fetchall()
# graph replays by duration cluster
durs = [(e - s) / 1e6 for s, e, _, _ in g]
print(f"graph replays {len(g)}; duration ms: min {min(durs):.3f} med {st.median(durs):.3f} max {max(durs):.3f}")
byid = collections.defaultdict(list)
for s, e, gid, sid in g: byid[gid].append((e - s) / 1e6)
for gid, d in sorted(byid.items(), key=lambda x: -len(x[1])): print(f"  graphId {gid}: n {len(d)} med {st.median(d):.3f} ms")
big = [x for x in g if (x[1] - x[0]) / 1e6 >= max(thr, 0.5 * st.median(durs))]
# graph-internal kernels have graphNodeId set (graph trace=graph still lists none); eager = kernels not covered by a graph interval
import bisect
gs = [x[0] for x in g]
def in_graph(t0, t1):
    i = bisect.bisect_right(gs, t0) - 1
    return i >= 0 and g[i][0] <= t0 and t1 <= g[i][1]
eager = [k for k in ks if not in_graph(k[0], k[1])]
print(f"kernels total {len(ks)}; eager (outside graph replay) {len(eager)}; memcpy {len(mc)}; memset {len(ms)}")
if len(big) < 2: sys.exit("no passes")
rows = []
for a, b in zip(big, big[1:]):
    t0, t1 = a[0], b[0]; period = (t1 - t0) / 1e6
    busy = (a[1] - a[0])
    gap0, gap1 = a[1], b[0]
    # gpu activity inside the gap
    act = [(k[0], k[1], 'k:' + k[2] if isinstance(k[2], str) else 'k:' + str(names.get(k[2], k[2]))) for k in eager if gap0 <= k[0] < gap1]
    act += [(m[0], m[1], f'memcpy{m[3]}:{m[2]}B') for m in mc if gap0 <= m[0] < gap1]
    act += [(m[0], m[1], 'memset') for m in ms if gap0 <= m[0] < gap1]
    other_g = [x for x in g if gap0 <= x[0] < gap1 and x is not b]
    act += [(x[0], x[1], 'graph') for x in other_g]
    act.sort()
    # union of activity in gap
    cov = 0; cur = gap0
    for s, e, _ in act:
        s = max(s, cur); 
        if e > s: cov += e - s; cur = e
    idle = (gap1 - gap0) - cov
    rts = [(r[0], r[1], names.get(r[2], str(r[2]))) for r in rt if gap0 <= r[0] < gap1]
    sys_ = [(r[0], r[1], r[2]) for r in sy if gap0 <= r[0] < gap1]
    rows.append(dict(period=period, graph=busy / 1e6, gap=(gap1 - gap0) / 1e6, gpu_in_gap=cov / 1e6, idle=idle / 1e6, nact=len(act), act=act, rts=rts, syncs=sys_, gap0=gap0, gap1=gap1))
def med(f): return st.median(r[f] for r in rows)
def mean(f): return sum(r[f] for r in rows) / len(rows)
print(f"\npasses {len(rows)} (graph >= {max(thr, 0.5*st.median(durs)):.2f} ms)")
for f in ('period', 'graph', 'gap', 'gpu_in_gap', 'idle'):
    print(f"  {f:11s} mean {mean(f):7.3f} ms  med {med(f):7.3f}  p10 {sorted(r[f] for r in rows)[len(rows)//10]:7.3f}  p90 {sorted(r[f] for r in rows)[9*len(rows)//10]:7.3f}")
print(f"  idle share of pass period: {100*sum(r['idle'] for r in rows)/sum(r['period'] for r in rows):.2f} %   (gap incl. eager GPU work: {100*sum(r['gap'] for r in rows)/sum(r['period'] for r in rows):.2f} %)")
print(f"  eager activity per gap: mean {mean('nact'):.1f}")
cnt = collections.Counter(); tot = collections.Counter()
for r in rows:
    for s, e, n in r['act']: cnt[n.split(':')[0] if n.startswith('memcpy') else n] += 1; tot[n.split(':')[0] if n.startswith('memcpy') else n] += (e - s)
print("\nGPU work in the gaps (name: count/pass, total us/pass):")
for n, c in cnt.most_common(12): print(f"  {n[:60]:60s} {c/len(rows):6.2f}  {tot[n]/len(rows)/1e3:8.2f} us")
rc = collections.Counter(); rtot = collections.Counter()
for r in rows:
    for s, e, n in r['rts']: rc[n] += 1; rtot[n] += (e - s)
print("\nCUDA runtime API calls in the gaps (name: count/pass, total us/pass on the host):")
for n, c in rc.most_common(14): print(f"  {n[:50]:50s} {c/len(rows):6.2f}  {rtot[n]/len(rows)/1e3:8.2f} us")
# where does the idle sit: first idle after graph end (host: download wait, accept, rollback, setup, launch)
r = rows[len(rows)//2]
print("\nmedian-ish pass timeline (us from previous graph end):")
t = r['gap0']
ev = [(s, e, 'GPU ' + n) for s, e, n in r['act']] + [(s, e, 'API ' + n) for s, e, n in r['rts']] + [(s, e, 'SYNC') for s, e, _ in r['syncs']]
for s, e, n in sorted(ev)[:60]: print(f"  +{(s-t)/1e3:8.1f} dur {(e-s)/1e3:8.1f}  {n[:70]}")
print(f"  next graph starts +{(r['gap1']-t)/1e3:.1f}")
