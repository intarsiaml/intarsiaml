#!/usr/bin/env python3
"""Dense in-engine tile sweep (eng3/k-<M>-<tile>-r<n>.jsonl, kernel mode): per-class GPU us per call (median over the 36 layers,
then over rounds) and the per-layer GEMM sum (2 qo + kv + gu + down) per forced tile.  python3 engdense.py [dir]"""
import glob, json, re, statistics, sys, collections
d = sys.argv[1] if len(sys.argv) > 1 else "eng3"
cls = {(4096, 4096): "qo", (2048, 4096): "kv", (24576, 4096): "gu", (4096, 12288): "down"}
data = collections.defaultdict(list)
tm = {}
for f in sorted(glob.glob(f"{d}/k-*-r*.jsonl")):
    m = re.fullmatch(rf"{d}/k-(\d+)-(\w+)-r(\d+)\.jsonl", f)
    if not m:
        continue
    size, tile, r = int(m.group(1)), m.group(2), int(m.group(3))
    per = collections.defaultdict(list)
    for l in open(f):
        x = json.loads(l)
        if x["op"] in ("matmul_bf16_nax", "matmul_pair_bf16_nax") and x["t"] > 1:
            c = cls.get((x["n"], x["k"]))
            if c:
                per[c].append((x["e"] - x["s"]) * 1e6)
                tm[size] = x["t"]
    for c, v in per.items():
        data[(size, tile, c)].append(statistics.median(v))
tiles = ["t128x64w4", "t64x64w4", "t64x64w2", "t32x128w2"]
short = {"t128x64w4": "128x64w4", "t64x64w4": "64x64w4", "t64x64w2": "64x64w2", "t32x128w2": "32x128w2"}
print("M | per-layer GEMM us per forced tile (" + " ".join(short[t] for t in tiles) + ") | best uniform | oracle per class (sum) | best tile per class (qo kv gu down)")
for size in sorted({k[0] for k in data}):
    cm = {(t, c): statistics.median(data[(size, t, c)]) for t in tiles for c in ("qo", "kv", "gu", "down") if data.get((size, t, c))}
    if len(cm) < 16:
        continue
    layer = {t: 2 * cm[(t, "qo")] + cm[(t, "kv")] + cm[(t, "gu")] + cm[(t, "down")] for t in tiles}
    bu = min(layer, key=layer.get)
    best = {c: min(tiles, key=lambda t: cm[(t, c)]) for c in ("qo", "kv", "gu", "down")}
    orc = 2 * cm[(best["qo"], "qo")] + sum(cm[(best[c], c)] for c in ("kv", "gu", "down"))
    print(f"{tm[size]} | " + " ".join(f"{layer[t]:7.0f}" for t in tiles) + f" | {short[bu]} {100*(layer[bu]/layer['t128x64w4']-1):+.1f}% | {orc:.0f} ({100*(orc/layer['t128x64w4']-1):+.1f}%) | " + " ".join(short[best[c]] for c in ("qo", "kv", "gu", "down")))
