#!/usr/bin/env python3
"""an5.py RUNDIR OUT.md : rows.json (from an4.py) summarised by tail class, raw / filtered (>=1450 MHz) / clock-scaled."""
import sys, json, statistics as st, collections
D = sys.argv[1]
rows = json.load(open(f"{D}/rows.json"))
def cls(r):
    l = r["last"]
    big = 200 if r["chunk"] == 256 else 2000
    return "a. tail >= 1 chunk-ish (%s+ rows)" % big if l >= big else ("b. tail 2-15" if l < 16 else ("c. tail 16-63" if l < 64 else "d. tail 64-%d" % (big - 1)))
g = collections.defaultdict(list)
for r in rows: g[(r["chunk"], cls(r), r["build"])].append(r)
out = ["| chunk | last-chunk class | build | n | median ms (all) | MHz | n fast (>=1450) | median ms (fast) | MHz (fast) | median ms scaled to 1600 MHz |", "|---|---|---|---|---|---|---|---|---|---|"]
for (c, k, b) in sorted(g):
    rs = g[(c, k, b)]; fast = [r for r in rs if r["mhz"] >= 1450]
    sc = [r["wall"] * r["mhz"] / 1600 for r in rs]
    out.append(f"| {c} | {k} | {b} | {len(rs)} | {st.median(r['wall'] for r in rs):.0f} | {st.mean(r['mhz'] for r in rs):.0f} | {len(fast)} | " +
               (f"{st.median(r['wall'] for r in fast):.0f} | {st.mean(r['mhz'] for r in fast):.0f}" if fast else "- | -") + f" | {st.median(sc):.0f} |")
open(sys.argv[2], "w").write("\n".join(out) + "\n")
