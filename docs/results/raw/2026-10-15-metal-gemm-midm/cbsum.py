#!/usr/bin/env python3
"""GPU busy (sum of command-buffer GPU time) per arm and prompt size from cb/*.jsonl: python3 cbsum.py <chunk>"""
import glob, json, re, statistics, sys, collections
chunk = sys.argv[1]
d = collections.defaultdict(list)
for f in sorted(glob.glob(f"cb/cb-*-c{chunk}-r*.jsonl")):
    m = re.fullmatch(rf"cb/cb-(\w+)-(\d+)-c{chunk}-r(\d+)(?:-\d+)?\.jsonl", f)
    if not m:
        continue
    arm, size, r = m.group(1), int(m.group(2)), int(m.group(3))
    rs = [json.loads(l) for l in open(f)]
    d[(size, arm)].append(sum(x["e"] - x["s"] for x in rs) * 1000)
sizes = sorted({k[0] for k in d})
print("size | " + " | ".join(f"{a} median (n) [min-max]" for a in ("base", "new")) + " | new vs base")
for s in sizes:
    row = []
    med = {}
    for a in ("base", "new"):
        v = d.get((s, a), [])
        if v:
            med[a] = statistics.median(v)
            row.append(f"{med[a]:.1f} ({len(v)}) [{min(v):.1f}-{max(v):.1f}]")
        else:
            row.append("-")
    nb = f"{(med['new']/med['base']-1)*100:+.1f}%" if "new" in med and "base" in med else "-"
    print(f"{s} | " + " | ".join(row) + f" | {nb}")
