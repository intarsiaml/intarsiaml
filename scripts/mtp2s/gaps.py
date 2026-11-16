#!/usr/bin/env python3
"""Lane mtp2s (2026-11-16): where the idle time of an eager pass sits. gaps.py <file.sqlite> [kernel_count=1464]

Segments are cut at every device-to-host copy (as seg.py). For the segments with the given kernel count, walk the kernels in start order
and attribute every stretch where no kernel was running (a gap on the union timeline) to the kernel that ends it (the one the host was
late with). Prints per kernel name: gap ms per pass, launches per pass, mean kernel ms, then the pass's gap total and the gap histogram."""
import collections, sqlite3, statistics as st, sys

db = sqlite3.connect(sys.argv[1]); want = int(sys.argv[2]) if len(sys.argv) > 2 else 1464
names = dict(db.execute("select id, value from StringIds"))
ks = [(s, e, names.get(sn, str(sn))) for s, e, sn in db.execute("select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start")]
cps = db.execute("select start, end from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 2 order by start").fetchall()
segs, ki = [], 0
for cs, ce in cps:
    cur = []
    while ki < len(ks) and ks[ki][0] < cs:
        cur.append(ks[ki]); ki += 1
    if cur and len(cur) == want:
        segs.append(cur)
gap_by, cnt_by, dur_by = collections.Counter(), collections.Counter(), collections.Counter()
tot, hist = [], collections.Counter()
for cur in segs:
    end, g = cur[0][1], 0
    for s, e, nm in cur[1:]:
        if s > end:
            gap_by[nm] += s - end; g += s - end
            hist["<5us" if s - end < 5e3 else "<20us" if s - end < 20e3 else "<100us" if s - end < 100e3 else ">=100us"] += s - end
        end = max(end, e)
        cnt_by[nm] += 1; dur_by[nm] += e - s
    tot.append(g)
n = len(segs)
print(f"{n} passes of {want} kernels; idle inside the pass (no kernel running), mean {st.mean(tot)/1e6:.3f} ms, median {st.median(tot)/1e6:.3f} ms")
print("gap time by size class (ms per pass):", {k: round(v / n / 1e6, 3) for k, v in hist.items()})
print("\ngap ms per pass attributed to the kernel that ends the gap (top 20): gap, launches per pass, mean kernel us")
for nm, g in gap_by.most_common(20):
    print(f"  {g / n / 1e6:7.3f}  {cnt_by[nm] / n:6.1f}  {dur_by[nm] / cnt_by[nm] / 1e3:8.1f}  {nm[:100]}")
