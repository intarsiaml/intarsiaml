#!/usr/bin/env python3
"""stampsum.py <stamps.txt>...: per op tag, the summed wall time from each op's first stamp (before its first dispatch) to
its last (after its last piece), from INTARSIA_HIP_STAMP files; plus the whole stamped span. Compares where an unprofiled
prefill spends its time between runs (e.g. split on vs off)."""
import sys
from collections import defaultdict

for path in sys.argv[1:]:
    with open(path) as f:
        khz = float(f.readline().split()[2])
        st = [(int(a) / (khz / 1000.0), t, int(p)) for a, t, p in (l.split() for l in f if l.strip())]
    tot, cnt, op0 = defaultdict(float), defaultdict(int), None
    for i, (ts, tag, piece) in enumerate(st):
        if piece < 0:
            op0 = ts
            cnt[tag] += 1
        nxt = st[i + 1] if i + 1 < len(st) else None
        if piece >= 0 and (nxt is None or nxt[2] < 0):
            tot[tag] += ts - op0
    span = (st[-1][0] - st[0][0]) / 1e3
    print(f"{path}: {len(st)} stamps, span {span:.1f} ms; " +
          ", ".join(f"{t} {cnt[t]} ops {tot[t] / 1e3:.2f} ms" for t in sorted(tot)) + f"; sum {sum(tot.values()) / 1e3:.2f} ms")
