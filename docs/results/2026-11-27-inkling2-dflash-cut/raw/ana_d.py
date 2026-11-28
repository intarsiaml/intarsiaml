#!/usr/bin/env python3
"""Lane ink-dflash-cut slot D: per-pass costs from INTARSIA_SPEC_TRACE=1 stderr (one file per arm).

Per pass the trace prints `spec: dflash2 draft flush .. block .. head .. selproj .. dl .. select .. total X ms`, then
`spec: token .. drafted D accepted A ..`, then `spec: mtp draft X ms, verify + rollback Y ms`. Prints per arm the mean draft phases,
and verify + rollback ms by drafted count (rows = D + 1), plus the whole-run tok/s.

usage: ana_d.py ARM.err ...
"""
import re, statistics as st, sys
from collections import defaultdict

PH = ("flush", "block", "head", "selproj", "dl", "select", "total")
for f in sys.argv[1:]:
    ph = defaultdict(list)
    ver = defaultdict(list)
    drafted = None
    overall = ""
    for l in open(f, errors="replace"):
        if l.startswith("spec: dflash2 draft flush"):
            w = l.split()
            for i, t in enumerate(w):
                if t in PH:
                    ph[t].append(float(w[i + 1]))
            continue
        m = re.match(r"spec: \S*\s*token \d+ pos \d+ drafted (\d+) accepted (\d+)", l)
        if m:
            drafted = int(m.group(1))
            continue
        m = re.match(r"spec: mtp draft ([0-9.]+) ms, verify \+ rollback ([0-9.]+) ms", l)
        if m and drafted is not None:
            ver[drafted].append(float(m.group(2)))
            drafted = None
            continue
        if l.startswith("decode overall"):
            overall = l.strip()
    n = sum(len(v) for v in ver.values())
    print(f"== {f.rsplit('/', 1)[-1]}  passes {n}  {overall}")
    print("   draft ms (median): " + " ".join(f"{k} {st.median(ph[k]):.3f}" for k in PH if ph[k]))
    allv = [x for v in ver.values() for x in v]
    print(f"   verify+rollback ms: median {st.median(allv):.2f}, mean {st.mean(allv):.2f}")
    print("   by drafted (rows-1): " + "  ".join(f"{d}: n{len(v)} med {st.median(v):.2f}" for d, v in sorted(ver.items())))
