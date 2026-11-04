#!/usr/bin/env python3
"""Sweep analysis: solo (C=2048) vs behind (C=256) per prompt, q4 at other busy chunks. rows before first id difference only, since context is shared there."""
import json, statistics as st
O = ".scratch/logs/sweep/"
REF = json.load(open("docs/results/2026-11-03-m27-hs-rowtable/raw/ident/def-mb2-ids.json"))
def rm(r): return {int(x[0]): x[1] for x in r}
def cmp(a, b):
    d = next((i for i, (x, y) in enumerate(zip(a["ids"], b["ids"])) if x != y), None)
    n = d if d is not None else len(a["ids"])
    n = min(n + 1, len(a["top"]), len(b["top"]))  # the diverging row itself shares its context
    t1, t5, kl, same, mingap = [], [], [], 0, 9
    for t in range(n):
        x, y = rm(a["top"][t]), rm(b["top"][t])
        if x == y: same += 1
        common = x.keys() & y.keys()
        top = sorted(x, key=x.get, reverse=True)
        t1.append(abs(x[top[0]] - y.get(top[0], x[top[0]])))
        t5.append(max(abs(x[i] - y[i]) for i in top[:5] if i in y))
        kl.append(sum(__import__("math").exp(x[i]) * (x[i] - y[i]) for i in common))
    return dict(first=d, rows=n, bit_equal_rows=same, top1_med=st.median(t1), top1_max=max(t1), top5_max=max(t5), kl_mean=st.mean(kl), kl_max=max(kl))
a = json.load(open(O + "all.json"))
print("prompt plen | ids: first diff solo vs behind | bit-equal rows | top1 |dlp| median, max | top5 max | KL mean, max | solo=ref behind=ref")
for q in range(10):
    s, b = a[f"q{q}-solo"], a[f"q{q}-behind"]
    c = cmp(s, b); r = REF[f"r1-q{q}"]["ids"]
    print(f"q{q} {s['plen']:6d} | {str(c['first']):5} | {c['bit_equal_rows']:3d}/{c['rows']:3d} | {c['top1_med']:.4f} {c['top1_max']:.4f} | {c['top5_max']:.4f} | {c['kl_mean']:.2e} {c['kl_max']:.2e} | {s['ids']==r} {b['ids']==r}")
print("\nq4 behind a decoding row at other busy chunks, and solo variants")
A = a["q4-solo"]; B = a["q4-behind"]
def g(v):
    m = rm(v["top"][184]); return m[14663], m[557], m[14663] - m[557]
print(f"{'solo-2048':10} id184={A['ids'][184]:<6} lp14663={g(A)[0]:.6f} lp557={g(A)[1]:.6f} gap={g(A)[2]:+.6f}")
print(f"{'behind-256':10} id184={B['ids'][184]:<6} lp14663={g(B)[0]:.6f} lp557={g(B)[1]:.6f} gap={g(B)[2]:+.6f}")
for t, k in [("bc32","q4-behind"),("bc64","q4-behind"),("bc128","q4-behind"),("bc512","q4-behind"),("bc771","q4-behind"),("pc256","q4-solo"),("mb1","q4-solo")]:
    v = json.load(open(O + t + ".json"))[k]
    c = cmp(v, A); c2 = cmp(v, B)
    print(f"{t:10} id184={v['ids'][184]:<6} lp14663={g(v)[0]:.6f} lp557={g(v)[1]:.6f} gap={g(v)[2]:+.6f} | vs solo: bit-equal {c['bit_equal_rows']}/{c['rows']} first-diff {c['first']} | vs 256: bit-equal {c2['bit_equal_rows']}/{c2['rows']} first-diff {c2['first']}")
for t in ["all","bc32","bc64","bc128","bc512","bc771","pc256","mb1"]:
    print(t, sorted({l.split("chunk: ")[1].split(",")[0]+"/"+l.split("decoding: ")[1].split(" ")[0] for l in open(O+t+"-server.err") if "PrefillStart" in l}))
