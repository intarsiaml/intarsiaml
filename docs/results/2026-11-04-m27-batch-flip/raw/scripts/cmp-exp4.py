#!/usr/bin/env python3
import json, os, sys
sys.argv = ["x"]; exec(open(".scratch/cmp-exp3.py").read().split("L = json.load")[0])
O = ".scratch/logs/exp4/"
n = json.load(open(O + "nax0.json")); c = cmp(n["q4-solo"], n["q4-behind"])
m = rm(n["q4-solo"]["top"][184]); print(f"NAX off, q4: solo vs behind first diff {c[0]} bit-equal {c[1]}/{c[2]} id184={n['q4-solo']['ids'][184]} gap={m.get(14663, float('nan')) - m.get(557, float('nan')):+.6f}")
for f in ("nax0-len", "peel16-len", "peel8-len", "nopeel-len"):
    print(f); L = json.load(open(O + f + ".json"))
    for k, v in L.items():
        d, s, nn, t5, km, kx = cmp(v["solo"], v["behind"]); print(f"  plen {k:5} tail {int(k) % 256:3d} first diff {str(d):5} bit-equal {s}/{nn} KL mean {km:.2e}")
