#!/usr/bin/env python3
"""Timeline of a cb trace around the long prefill buffers: prints every command buffer longer than MS ms and the gaps between buffers longer than MS ms.
usage: cbgap.py trace.jsonl [ms=5]"""
import json, sys
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip().startswith("{")]
ms = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
t0 = recs[0]["s"]; end = None
for r in recs:
    if end is not None and (r["s"] - end) * 1e3 > ms: print(f"   gap {1e3*(r['s']-end):8.1f} ms (GPU idle) before cb {r['i']}")
    d = (r["e"] - r["s"]) * 1e3
    if d > ms: print(f"cb {r['i']:5d} at {1e3*(r['s']-t0):9.1f} ms: {d:8.1f} ms  op {r['op']} t={r['t']} nd={r['nd']}")
    end = r["e"] if end is None else max(end, r["e"])
