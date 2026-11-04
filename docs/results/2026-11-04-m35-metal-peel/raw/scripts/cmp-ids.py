#!/usr/bin/env python3
"""usage: cmp-ids.py IDS.json OUTDIR MOD  -- table of main vs peel, solo vs behind, with the chunk schedule from the trace.
MOD: the busy-chunk-exposure modulus text only used for the 'exposed' column (35: p%4096 in 2050..2057, 27: p%256 in 2..9)."""
import json, re, sys
d = json.load(open(sys.argv[1])); out = sys.argv[2]; mod = sys.argv[3]
keys = sorted({k.split("/")[1] for k in d}, key=int)
tags = sorted({k.split("/")[0] for k in d})
def sched(tag):
    ev = {}
    for l in open(f"{out}/server-{tag}.err"):
        m = re.search(r"(PrefillStart|Chunk) \{ iter: (\d+), id: (\d+), (.*) \}", l)
        if not m: continue
        e = ev.setdefault(int(m[3]), {"c": []}); kv = dict(x.split(": ") for x in m[4].split(", "))
        if m[1] == "PrefillStart": e["C"], e["dec"] = int(kv["chunk"]), int(kv["decoding"])
        else: e["c"].append(int(kv["len"]))
    by = {}
    for i, e in sorted(ev.items()):
        if "C" in e: by.setdefault(sum(e["c"]), []).append((e["C"], e["dec"], e["c"]))
    return by
S = {t: sched(t) for t in tags if __import__("os").path.exists(f"{out}/server-{t}.err")}
exposed = (lambda p: 2050 <= p % 4096 <= 2057) if mod == "35" else (lambda p: 2 <= p % 256 <= 9 and p > 256 and p % 2048 >= 256)
fd = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
def sc(t, p, i):
    v = S.get(t, {}).get(p, []); 
    if i < len(v): c = v[i][2]; return (f"{c[:-1][-1] if len(c) > 1 else ''}+{c[-1]}" if len(c) > 1 else f"{c[0]}") + ("" if len(c) < 3 else f" ({len(c)} passes)")
    return "?"
print("plen mod256 mod4096 exp | " + " | ".join(f"{t}: solo sched / behind sched / solo==behind" for t in tags) + " | peel==main solo / behind")
for k in keys:
    p = int(k); g = lambda t, m: d.get(f"{t}/{k}/{m}", {}).get("ids", [])
    cols = []
    for t in tags:
        a, b = g(t, "solo"), g(t, "behind")
        cols.append(f"{sc(t, p, 0)} / {sc(t, p, 1)} / " + ("EQ" if a == b else f"DIFF@{fd(a, b)}"))
    cmpmb = ""
    if "main" in tags and "peel" in tags:
        cmpmb = ("EQ" if g("peel", "solo") == g("main", "solo") else f"DIFF@{fd(g('peel','solo'), g('main','solo'))}") + " / " + ("EQ" if g("peel", "behind") == g("main", "behind") else f"DIFF@{fd(g('peel','behind'), g('main','behind'))}")
    print(f"{p:5d} {p%256:3d} {p%4096:4d} {'Y' if exposed(p) else '-'} | " + " | ".join(cols) + " | " + cmpmb)
