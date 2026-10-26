#!/usr/bin/env python3
"""Lane mpf: prefill tok/s, TTFT and decode tok/s of scripts/mpf/longctx.sh windows, per prompt size and arm, over all rounds, and each arm's ratio to off.
usage: python3 -I tools/mpf/lc_table.py <longctx dir> [more dirs ...]   (files <arm>-r<N>.speed.txt; prompt sizes are bucketed 4k / 16k / 34k)
Rows come from localeval speed's table: | prompt | ttft s | prefill tok/s (min-max) | - | decode tok/s (min-max) | ...
The table's TTFT has two decimals (1.7 % steps at 0.6 s); localeval defines prefill tok/s = uncached prompt tokens / TTFT, so the printed TTFT is prompt / prefill."""
import glob, os, re, statistics as st, sys
rows = {}
for d in sys.argv[1:]:
    for f in sorted(glob.glob(os.path.join(d, "*.speed.txt"))):
        arm, rnd = re.match(r"(.+)-r(\d+)\.speed\.txt", os.path.basename(f)).groups()
        for line in open(f):
            m = re.match(r"\| (\d+) \| ([\d.]+) \| (\d+) \(\d+.\d+\) \| \S+ \| ([\d.]+) \(", line)
            if not m: continue
            n = int(m.group(1)); b = "4k" if n < 8000 else "16k" if n < 25000 else "34k"
            rows.setdefault((b, arm), []).append((int(rnd), float(m.group(3)), n / float(m.group(3)), float(m.group(4)), n))
print(f"{'size':5} {'arm':5} {'rounds':>6} {'prefill tok/s':>14} {'x off':>7} {'TTFT s (n/prefill)':>19} {'x off':>7} {'decode tok/s':>13} {'x off':>7}")
for b in ("4k", "16k", "34k"):
    off = rows.get((b, "off"))
    for arm in sorted({a for (bb, a) in rows if bb == b}, key=lambda a: (a != "off", a)):
        r = rows[(b, arm)]
        med = lambda i: st.median(x[i] for x in r)
        o = lambda i: st.median(x[i] for x in off)
        print(f"{b:5} {arm:5} {len(r):6d} {med(1):14.0f} {med(1) / o(1):7.3f} {med(2):19.3f} {med(2) / o(2):7.3f} {med(3):13.1f} {med(3) / o(3):7.3f}")
