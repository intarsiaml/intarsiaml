#!/usr/bin/env python3
"""In-engine A/B of pick arms (eng*/k-<M>-<arm>-r<n>.jsonl, kernel mode): per-layer GEMM GPU us (2 qo + kv + gu + down; per-class
median over the 36 layers of a run, then median over rounds) with [min..max over rounds].  python3 enga.py <dir> [arm ...]
first arm is the baseline."""
import glob, json, re, statistics, sys, collections
d = sys.argv[1]
cls = {(4096, 4096): "qo", (2048, 4096): "kv", (24576, 4096): "gu", (4096, 12288): "down"}
data = collections.defaultdict(list); tm = {}
for f in sorted(glob.glob(f"{d}/k-*-r*.jsonl")):
    m = re.fullmatch(rf"{d}/k-(\d+)-(\w+)-r(\d+)\.jsonl", f)
    if not m:
        continue
    size, arm, r = int(m.group(1)), m.group(2), int(m.group(3))
    for l in open(f):
        x = json.loads(l)
        if x["op"] in ("matmul_bf16_nax", "matmul_pair_bf16_nax") and x["t"] > 1:
            c = cls.get((x["n"], x["k"]))
            if c:
                data[(size, arm, c, r)].append((x["e"] - x["s"]) * 1e6); tm[size] = x["t"]
arms = sys.argv[2:] or sorted({k[1] for k in data})
print("M | " + " | ".join(f"{a} us/layer [min..max]" for a in arms) + " | change vs " + arms[0])
tot = collections.defaultdict(float)
for size in sorted({k[0] for k in data}, key=lambda s: tm[s]):
    layer = {}
    for a in arms:
        rounds = sorted({k[3] for k in data if k[0] == size and k[1] == a})
        per = [sum(mult * statistics.median(data[(size, a, c, r)]) for c, mult in (("qo", 2), ("kv", 1), ("gu", 1), ("down", 1))) for r in rounds]
        layer[a] = (statistics.median(per), min(per), max(per)); tot[a] += layer[a][0]
    print(f"{tm[size]} | " + " | ".join(f"{layer[a][0]:.0f} [{layer[a][1]:.0f}..{layer[a][2]:.0f}]" for a in arms) + " | " + " ".join(f"{100*(layer[a][0]/layer[arms[0]][0]-1):+.1f}%" for a in arms[1:]))
print("sum over sizes | " + " | ".join(f"{tot[a]:.0f}" for a in arms) + " | " + " ".join(f"{100*(tot[a]/tot[arms[0]]-1):+.1f}%" for a in arms[1:]))
