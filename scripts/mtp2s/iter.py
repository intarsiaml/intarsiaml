#!/usr/bin/env python3
"""Lane mtp2s: per-iteration timeline of a batched-verify window (nsys sqlite). iter.py <file.sqlite> [verify_kernels=1464]
An iteration = from one batched verify pass's first kernel to the next one's. Reports the median period and, inside it, the verify
span, the draft passes' spans, the device-to-host copies (count, bytes, time) and the time the GPU is idle (no kernel, no copy)."""
import sqlite3, statistics as st, sys
db = sqlite3.connect(sys.argv[1]); nver = int(sys.argv[2]) if len(sys.argv) > 2 else 1464
ks = db.execute("select start, end from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
cps = db.execute("select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where copyKind = 2 order by start").fetchall()
# segments between D2H copies (as seg.py)
segs, ki = [], 0
for cs, ce, nb in cps:
    cur = []
    while ki < len(ks) and ks[ki][0] < cs:
        cur.append(ks[ki]); ki += 1
    if cur:
        segs.append((cur[0][0], ce, len(cur)))
ver = [s for s in segs if abs(s[2] - nver) <= 4]
rows = []
for a, b in zip(ver, ver[1:]):
    t0, t1 = a[0], b[0]
    if t1 - t0 > 80e6:
        continue  # a prefill or a tail in between
    inner = [s for s in segs if t0 <= s[0] < t1]
    kin = [k for k in ks if t0 <= k[0] < t1]
    cin = [c for c in cps if t0 <= c[0] < t1]
    busy, cend = 0, t0
    for s_, e_ in sorted([(k[0], k[1]) for k in kin] + [(c[0], c[1]) for c in cin]):
        s2 = max(s_, cend)
        if e_ > s2:
            busy += e_ - s2; cend = e_
    rows.append(dict(period=t1 - t0, verify=a[1] - a[0], drafts=sum(s[1] - s[0] for s in inner[1:]), ndraft=len(inner) - 1,
                     copies=len(cin), cbytes=sum(c[2] for c in cin), ctime=sum(c[1] - c[0] for c in cin), idle=(t1 - t0) - busy))
m = lambda k: st.median(r[k] for r in rows)
print(f"{sys.argv[1]}: {len(rows)} iterations; medians: period {m('period')/1e6:.2f} ms, verify span {m('verify')/1e6:.2f}, draft passes {m('ndraft'):.0f} "
      f"spanning {m('drafts')/1e6:.2f}, D2H copies {m('copies'):.0f} ({m('cbytes')/1e6:.2f} MB, {m('ctime')/1e6:.3f} ms), GPU idle {m('idle')/1e6:.2f} ms "
      f"(no kernel or copy running)")
