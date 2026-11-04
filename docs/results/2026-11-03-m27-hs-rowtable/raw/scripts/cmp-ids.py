#!/usr/bin/env python3
"""Compare the per-request greedy token ids of the identity runs: rt vs def at each max-batch, and every arm vs def-mb2."""
import json, sys, os
d = sys.argv[1]
L = lambda t: json.load(open(f"{d}/{t}-ids.json"))
ref = L("def-mb2")
bad = 0
def cmp(a, b, na, nb):
    global bad
    ks = sorted(a)
    same = [k for k in ks if a[k]["ids"] == b[k]["ids"] and a[k]["ids"]]
    diff = [k for k in ks if a[k]["ids"] != b[k]["ids"]]
    bad += len(diff)
    print(f"{na} vs {nb}: {len(same)}/{len(ks)} requests identical ids ({sum(len(a[k]['ids']) for k in same)} ids)" + (f"  DIFF: {diff}" if diff else ""))
for mb in (2, 4, 8):
    r, b = L(f"rt-mb{mb}"), L(f"def-mb{mb}")
    cmp(r, b, f"rt-mb{mb}", f"def-mb{mb}")
    cmp(r, ref, f"rt-mb{mb}", "def-mb2")
    cmp(b, ref, f"def-mb{mb}", "def-mb2")
for mb in (2, 4, 8):
    for t in ("def", "rt"):
        print(t, mb, open(f"{d}/{t}-mb{mb}-batch-counters.txt").read().replace("\n", " ")[:600])
print("TOTAL DIFFERING REQUESTS:", bad)
sys.exit(1 if bad else 0)
