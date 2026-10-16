#!/usr/bin/env python3
"""Cold `generate` runs (cold.sh): per arm and chunk, prefill wall (engine), GPU busy, span, idle in span, lead (first record start is not
comparable across processes, so lead is not reported), cb count. python3 coldsum.py <dir>"""
import glob, json, re, sys, statistics, collections
d = sys.argv[1]
rows = collections.defaultdict(list)
for f in sorted(glob.glob(f"{d}/*.trace.jsonl")):
    m = re.fullmatch(r".*/(\w+)-r(\d+)-(\d+)-c(\d+)\.trace\.jsonl", f)
    arm, r, size, c = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    rs = [json.loads(l) for l in open(f)]
    busy = sum(x["e"] - x["s"] for x in rs) * 1000
    span = (rs[-1]["e"] - rs[0]["s"]) * 1000
    wall = float(re.search(r"prefill (\d+) ms", open(f.replace(".trace.jsonl", ".err")).read()).group(1))
    rows[(c, arm)].append((wall, busy, span, span - busy, len(rs)))
for (c, arm), v in sorted(rows.items()):
    med = lambda i: statistics.median(x[i] for x in v)
    print(f"chunk {c:5d} {arm:12s} n {len(v)} prefill wall {med(0):7.1f} [{min(x[0] for x in v):.0f}..{max(x[0] for x in v):.0f}] busy {med(1):7.1f} span {med(2):7.1f} idle-in-span {med(3):5.1f} cbs {med(4):.0f}")
