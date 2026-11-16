#!/usr/bin/env python3
"""Lane msmallm: histogram of the M (rows) of every MXFP4 matmul call in an INTARSIA_METAL_PROFILE=cb trace taken with
INTARSIA_METAL_COMMIT_EVERY=1 (one record per dispatch; `<op>#1`, `#2` are the later dispatches of the same call and are
skipped). usage: mhist.py trace.jsonl[.gz] ..."""
import collections, gzip, json, sys

for f in sys.argv[1:]:
    c = collections.Counter()
    for l in gzip.open(f, "rt") if f.endswith(".gz") else open(f):
        if not l.startswith("{"):
            continue
        r = json.loads(l)
        op = r["op"]
        if op.startswith("matmul_mxfp4") and "#" not in op:
            c[(op, r["t"])] += 1
    print(f"## {f.split('/')[-1]}")
    for (op, t), v in sorted(c.items()):
        print(f"  {op:24s} M={t:<6d} calls={v}")
