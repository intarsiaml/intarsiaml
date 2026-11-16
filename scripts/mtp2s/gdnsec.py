#!/usr/bin/env python3
"""Lane mtp2s (2026-11-16): the GatedDeltaNet share of an eager verify pass, averaged over ALL passes of a kernel count.
  gdnsec.py <file.sqlite> [kernel_count=1464] [gdn kernel name prefix = gdn_]
Segments are cut at every device-to-host copy (as seg.py). Per pass: the GDN section of each layer = from its first to its last
gdn_* kernel (the rows' conv / delta / conv-state launches and the slot copies between them), summed over the layers; the device-to-device
copies (state saves) in the pass and their time; the pass span. Prints mean / median over the passes."""
import sqlite3, statistics as st, sys

db = sqlite3.connect(sys.argv[1]); want = int(sys.argv[2]) if len(sys.argv) > 2 else 1464; pre = sys.argv[3] if len(sys.argv) > 3 else "gdn_"
names = dict(db.execute("select id, value from StringIds"))
ks = [(s, e, names.get(sn, "?")) for s, e, sn in db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
allc = db.execute("select start, end, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY order by start").fetchall()
d2h = [c for c in allc if c[2] == 2]
rows, ki = [], 0
for cs, ce, _ in d2h:
    cur = []
    while ki < len(ks) and ks[ki][0] < cs:
        cur.append(ks[ki]); ki += 1
    if len(cur) != want:
        continue
    t0 = cur[0][0]
    g = [k for k in cur if k[2].startswith(pre)]
    lay, c = [], [g[0]] if g else []
    for k in g[1:]:
        if k[0] - c[-1][1] > 100e3:
            lay.append(c); c = [k]
        else:
            c.append(k)
    if c:
        lay.append(c)
    cp = [x for x in allc if t0 <= x[0] <= ce and x[2] == 8]
    rows.append(dict(span=ce - t0, sec=sum(l[-1][1] - l[0][0] for l in lay), nl=len(lay), ncopy=len(cp), copy_t=sum(x[1] - x[0] for x in cp), gk=sum(k[1] - k[0] for k in g), ng=len(g)))
f = lambda k, d=1e6: (st.mean(r[k] for r in rows) / d, st.median(r[k] for r in rows) / d)
print(f"{sys.argv[1]}: {len(rows)} passes of {want} kernels")
print(f"  span ms {f('span')[0]:.3f} (median {f('span')[1]:.3f})")
print(f"  GDN sections ms {f('sec')[0]:.3f} (median {f('sec')[1]:.3f}) over {st.mean(r['nl'] for r in rows):.0f} layers; gdn kernels {st.mean(r['ng'] for r in rows):.0f} launches, {f('gk')[0]:.3f} ms of kernel time")
print(f"  D2D copies {st.mean(r['ncopy'] for r in rows):.0f} per pass, {f('copy_t')[0]:.3f} ms")
