#!/usr/bin/env python3
import json, math, re
def rm(r): return {int(x[0]): x[1] for x in r}
L = json.load(open(".scratch/logs/gates/35b-len2.json"))
for k, v in L.items():
    a, b = v["solo"], v["behind"]; d = next((i for i, (x, y) in enumerate(zip(a["ids"], b["ids"])) if x != y), None); n = (d + 1) if d is not None else 200
    same = sum(a["top"][t] == b["top"][t] for t in range(n)); K = []
    for t in range(n):
        x, y = rm(a["top"][t]), rm(b["top"][t]); K.append(sum(math.exp(x[i]) * (x[i] - y[i]) for i in x.keys() & y.keys()))
    print(k, "plen", a["plen"], "mod2048", a["plen"] % 2048, "first id diff", d, "bit-equal", same, "/", n, f"KL mean {sum(K) / n:.2e} max {max(K):.3f}")
ev = {}
for l in open(".scratch/logs/gates/35b2-server.err"):
    m = re.search(r"(PrefillStart|Chunk) \{ iter: (\d+), id: (\d+), (.*) \}", l)
    if not m: continue
    e = ev.setdefault(int(m[3]), {"c": []}); kv = dict(x.split(": ") for x in m[4].split(", "))
    if m[1] == "PrefillStart": e["C"], e["dec"] = kv["chunk"], kv["decoding"]
    else: e["c"].append(int(kv["len"]))
seen = {}
for i, e in sorted(ev.items()):
    p = sum(e["c"])
    if str(p) in L: seen.setdefault(p, set()).add((e["C"], e["dec"], str(e["c"])))
for p, s in sorted(seen.items()): print(p, sorted(s))
