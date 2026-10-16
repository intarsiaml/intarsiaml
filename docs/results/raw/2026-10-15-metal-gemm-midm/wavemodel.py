#!/usr/bin/env python3
"""Check a wave model (tile time vs threadgroup count from the cache-resident probe, waves1.txt) against the in-engine dense sweep
(eng3): does the model's tile choice have low regret?  python3 wavemodel.py"""
import re, json, glob, math, statistics, collections, bisect
probe = collections.defaultdict(list)
for l in open("waves1.txt"):
    if l.startswith("WAVES"):
        x = dict(t.split("=") for t in l.split()[1:])
        probe[x["tile"]].append((int(x["tgs"]), float(x["us"])))
tiles = {"128x64w4": (128, 64), "64x64w4": (64, 64), "64x64w2": (64, 64), "32x128w2": (32, 128)}
def f(tile, T):
    pts = sorted(probe[tile]); xs = [p[0] for p in pts]
    if T <= xs[0]: return pts[0][1]
    if T >= xs[-1]: return pts[-1][1] * T / xs[-1]
    i = bisect.bisect_left(xs, T)
    (x0, y0), (x1, y1) = pts[i - 1], pts[i]
    return y0 + (y1 - y0) * (T - x0) / (x1 - x0)
cls = {"qo": (4096, 4096, 1), "kv": (1024, 4096, 2), "gu": (12288, 4096, 2), "down": (4096, 12288, 1)}
# measured
d = "eng3"
cls_key = {(4096, 4096): "qo", (2048, 4096): "kv", (24576, 4096): "gu", (4096, 12288): "down"}
data = collections.defaultdict(list); tm = {}
for fn in sorted(glob.glob(f"{d}/k-*-r*.jsonl")):
    m = re.fullmatch(rf"{d}/k-(\d+)-(\w+)-r(\d+)\.jsonl", fn)
    if not m: continue
    size, tile = int(m.group(1)), m.group(2)[1:]
    per = collections.defaultdict(list)
    for l in open(fn):
        x = json.loads(l)
        if x["op"] in ("matmul_bf16_nax", "matmul_pair_bf16_nax") and x["t"] > 1:
            c = cls_key.get((x["n"], x["k"]))
            if c: per[c].append((x["e"] - x["s"]) * 1e6); tm[size] = x["t"]
    for c, v in per.items(): data[(size, tile, c)].append(statistics.median(v))
meas = {k: statistics.median(v) for k, v in data.items()}
def pred(tile, M, c):
    n, k, npair = cls[c]
    bm, bn = tiles[tile]
    T = math.ceil(M / bm) * (math.ceil(n / bn) * npair)
    return f(tile, T) * k / 4096
mult = {"qo": 2, "kv": 1, "gu": 1, "down": 1}
tot_reg = tot_base = tot_best = 0
print("M | model pick per class (qo kv gu down) | layer us: model-pick / 128x64w4 / measured-oracle | regret vs oracle")
for size in sorted({k[0] for k in meas}):
    M = tm[size]
    pick = {}
    for c in cls:
        pick[c] = min(tiles, key=lambda t: pred(t, M, c))
    got = sum(mult[c] * meas[(size, pick[c], c)] for c in cls)
    base = sum(mult[c] * meas[(size, "128x64w4", c)] for c in cls)
    orc = sum(mult[c] * min(meas[(size, t, c)] for t in tiles) for c in cls)
    tot_reg += got; tot_base += base; tot_best += orc
    print(f"{M} | {' '.join(pick[c] for c in cls)} | {got:.0f} / {base:.0f} / {orc:.0f} | {100*(got/orc-1):+.1f}%  (vs 128x64: {100*(got/base-1):+.1f}%)")
print(f"sum over sizes: model {tot_reg:.0f}, 128x64w4 {tot_base:.0f} ({100*(tot_reg/tot_base-1):+.1f}%), oracle {tot_best:.0f} ({100*(tot_best/tot_base-1):+.1f}%)")
