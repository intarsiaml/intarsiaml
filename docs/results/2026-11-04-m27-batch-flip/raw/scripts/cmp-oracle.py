#!/usr/bin/env python3
"""f64 reference vs each Metal path, rows where reference and path share context (same ids so far)."""
import json, math, statistics as st
def rm(r): return {int(x[0]): x[1] for x in r}
LEN = json.load(open(".scratch/logs/exp3/len.json")); LEN.update(json.load(open(".scratch/logs/exp4/nopeel-len.json")))
P = json.load(open(".scratch/logs/oracle/perturb.json")); F = json.load(open(".scratch/logs/oracle/ref-free.json"))
def kl(x, y):
    return sum(math.exp(x[i]) * (x[i] - y[i]) for i in x.keys() & y.keys())
for k in ["258", "260", "262", "1028"]:
    f = F.get(k)
    if not f: print(k, "no reference"); continue
    paths = {"solo-NAX": LEN[k]["solo"], "busy": LEN[k]["behind"], "allGEMV": P.get(f"nax0/{k}"), "tail3": P.get(f"tail3/{k}"), "tail5": P.get(f"tail5/{k}"), "tail7": P.get(f"tail7/{k}")}
    print(f"plen {k}: reference {len(f['ids'])} rows, wall {f['wall']:.0f} s")
    for n, v in paths.items():
        if not v: continue
        d = next((i for i, (a, b) in enumerate(zip(f["ids"], v["ids"])) if a != b), None)
        m = min((d + 1) if d is not None else len(f["ids"]), len(v["top"]), len(f["top"]))
        K = [kl(rm(f["top"][t]), rm(v["top"][t])) for t in range(m)]
        print(f"   {n:9} first id diff vs ref {str(d):5} rows {m:3d}  KL(ref||path) median {st.median(K):.2e} mean {st.mean(K):.2e} max {max(K):.3f}@{K.index(max(K))}  row0 {K[0]:.2e}")
    p = [(n, v) for n, v in paths.items() if v]
    n0 = min(len(f["top"]), min(len(v["top"]) for _, v in p))
    # path-to-path dispersion for comparison: solo vs busy on shared rows
    a, b = LEN[k]["solo"], LEN[k]["behind"]; d = next((i for i, (x, y) in enumerate(zip(a["ids"], b["ids"])) if x != y), None); m = (d + 1) if d is not None else 200
    K = [kl(rm(a["top"][t]), rm(b["top"][t])) for t in range(m)]
    print(f"   solo vs busy (path-to-path)     rows {m:3d}  KL median {st.median(K):.2e} mean {st.mean(K):.2e} max {max(K):.3f}")
    rows = {"258": [100, 104, 105], "260": [138, 197], "262": [188], "1028": [63]}[k]
    for r in rows:
        if r >= len(f["top"]): print("   row", r, "beyond reference"); continue
        same = f["ids"][:r] == LEN[k]["solo"]["ids"][:r]
        o = rm(f["top"][r]); top = sorted(rm(LEN[k]["solo"]["top"][r]), key=rm(LEN[k]["solo"]["top"][r]).get, reverse=True)[:2]
        print(f"   row {r} ref same-context={same} ref top {f['ids'][r]} " + " ".join(f"ref lp[{i}]={o.get(i, float('nan')):.4f}" for i in top))
