#!/usr/bin/env python3
"""8B non-recurrent control: f32 vs f64 CPU kernels. Same scoring as cmp-oracle2.py's 27B f32-vs-f64 lines."""
import json, math, statistics as st
def rm(r): return {int(x[0]): x[1] for x in r}
O = ".scratch/logs/oracle/"
a = json.load(open(O + "8b-f64.json")); b = json.load(open(O + "8b-f32.json"))
for k in a:
    if k not in b: continue
    f, c = a[k], b[k]; d = next((i for i, (x, y) in enumerate(zip(f["ids"], c["ids"])) if x != y), None); m = (d + 1) if d is not None else 200
    K, D, same = [], [], 0
    for t in range(m):
        x, y = rm(f["top"][t]), rm(c["top"][t]); same += x == y
        K.append(sum(math.exp(x[i]) * (x[i] - y[i]) for i in x.keys() & y.keys())); top = sorted(x, key=x.get, reverse=True)[:3]; D.append(max(abs(x[i] - y[i]) for i in top if i in y))
    print(f"8B plen {f['plen']} first id diff {d} rows {m} bit-equal {same} KL median {st.median(K):.1e} mean {st.mean(K):.1e} max {max(K):.3f} dlp>1 rows {sum(x > 1 for x in D)} row0 {D[0]:.4f} 50-row medians {[round(st.median(D[l:l + 50]), 3) for l in range(0, m, 50)]}")
