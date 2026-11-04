#!/usr/bin/env python3
import json, math, os
O = ".scratch/logs/exp3/"
def rm(r): return {int(x[0]): x[1] for x in r}
def cmp(a, b):
    d = next((i for i, (x, y) in enumerate(zip(a["ids"], b["ids"])) if x != y), None)
    n = min((d + 1) if d is not None else len(a["ids"]), len(a["top"]), len(b["top"]))
    same = t5 = 0; kls = []; t5 = 0.0
    for t in range(n):
        x, y = rm(a["top"][t]), rm(b["top"][t]); same += x == y
        top = sorted(x, key=x.get, reverse=True)[:5]
        t5 = max([t5] + [abs(x[i] - y[i]) for i in top if i in y])
        kls.append(sum(math.exp(x[i]) * (x[i] - y[i]) for i in x.keys() & y.keys()))
    return d, same, n, t5, sum(kls) / n, max(kls)
L = json.load(open(O + "len.json"))
print("plen  tail(mod256) solo-pass-tail | first id diff | bit-equal rows | top5 max dlp | KL mean, max")
for k, v in L.items():
    p = int(k); d, s, n, t5, km, kx = cmp(v["solo"], v["behind"])
    print(f"{p:5d} {p % 256:4d} {p % 2048:5d} | {str(d):5} | {s:3d}/{n:3d} | {t5:.4f} | {km:.2e} {kx:.2e}")
A = json.load(open(".scratch/logs/sweep/all.json")); S, B = A["q4-solo"], A["q4-behind"]
print("\nq4 (772), last pass t rows (body t-1), vs solo one-pass and vs 256 schedule")
for t in [6, 9, 10, 17, 18, 33, 34]:
    f = f"{O}t{t}.json"
    if not os.path.exists(f): print(t, "missing"); continue
    v = json.load(open(f))["q4-behind"]; c1, c2 = cmp(v, S), cmp(v, B)
    fd = next((i for i, (x, y) in enumerate(zip(v["ids"], B["ids"])) if x != y), None); m = rm(v["top"][184])
    print(f"t={t:2d} id184={v['ids'][184]:<6} first-diff-vs-256={fd} gap={m.get(14663, float('nan')) - m.get(557, float('nan')):+.6f} | vs solo bit-equal {c1[1]}/{c1[2]} | vs C=256 bit-equal {c2[1]}/{c2[2]}")
print("\npeel 16")
P = json.load(open(O + "peel16.json"))
for q in (3, 4, 5):
    s, b = P[f"q{q}-solo"], P[f"q{q}-behind"]; r = A[f"q{q}-behind"]; rs = A[f"q{q}-solo"]
    c = cmp(s, b); print(f"q{q}: peel solo vs peel behind first diff {c[0]} bit-equal {c[1]}/{c[2]} | peel behind == no-peel behind: {b['top'] == r['top']} | peel solo == no-peel solo: {s['top'] == rs['top']}")
