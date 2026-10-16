#!/usr/bin/env python3
"""Gap structure of a cb-mode trace: python3 gaps.py trace.jsonl. Prints busy / span / idle, idle split by gap size, and gaps by position in the 3-buffer-per-chunk pattern."""
import json, sys, collections
for f in sys.argv[1:]:
    rs = [json.loads(l) for l in open(f)]
    busy = sum(r["e"] - r["s"] for r in rs) * 1000
    span = (rs[-1]["e"] - rs[0]["s"]) * 1000
    gaps = [(rs[i]["s"] - rs[i - 1]["e"]) * 1000 for i in range(1, len(rs))]
    big = [g for g in gaps if g > 0.3]
    print(f"{f}: cbs {len(rs)} busy {busy:.1f} span {span:.1f} idle {span - busy:.1f} ms; gaps>0.3ms: {len(big)} sum {sum(big):.1f}; gaps<=0.3: sum {sum(g for g in gaps if g <= 0.3):.2f}")
    print("  dispatches per cb:", collections.Counter(r["nd"] for r in rs).most_common(6))
    print("  gap list (ms):", " ".join(f"{g:.2f}" for g in gaps[:40]))
