#!/usr/bin/env python3
"""Per-layer GEMM time and 4k-prompt GEMM time from SWEEP lines (shipped-at-the-time pick, picked=1): python3 layer.py sweepN.txt [tilepref]"""
import sys, collections
t = collections.defaultdict(dict)
for l in open(sys.argv[1]):
    if not l.startswith("SWEEP"): continue
    d = dict(x.split("=") for x in l.split()[1:])
    if d["picked"] == "1":
        t[int(d["M"])][d["shape"]] = float(d["us"])
def layer(m):  # q, o (2x qo), k+v pair, gate+up pair, down
    s = t[m]
    return 2 * s["qo"] + s["kv"] + s["gu"] + s["down"]
for m in sorted(t):
    if all(k in t[m] for k in ("qo", "kv", "gu", "down")):
        print(f"M={m}: per layer {layer(m)/1000:.3f} ms (qo {t[m]['qo']:.0f} x2, kv {t[m]['kv']:.0f}, gu {t[m]['gu']:.0f}, down {t[m]['down']:.0f} us)")
if 256 in t and 2048 in t:
    a = layer(256) * 36 * 16 / 1000
    b = layer(2048) * 36 * 2 / 1000
    print(f"4k prompt, 36 layers: 16 x M=256 = {a:.0f} ms, 2 x M=2048 = {b:.0f} ms, ratio {a/b:.3f}")
