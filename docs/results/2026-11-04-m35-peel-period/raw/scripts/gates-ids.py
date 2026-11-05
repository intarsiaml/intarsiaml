#!/usr/bin/env python3
"""usage: gates-ids.py OUTDIR IDS.json -- gates 1 and 2 of docs/results/2026-11-04-m35-peel-period.md, fixed before data.
Gate 1: exposed lengths, peel build: solo == behind, 256 ids, with the trace showing the behind row's first chunk planned with decoding >= 1.
Gate 2: the other lengths: peel == main, solo and behind, 256 ids (and the behind row overlapped)."""
import json, re, sys
out, idsf = sys.argv[1], sys.argv[2]
d = json.load(open(idsf))
G1 = [2050, 2052, 2055, 2056, 2057, 6146, 6153]
G2 = [258, 260, 265, 516, 772, 1028, 1540, 4356, 2058, 2064, 2300, 6154, 48, 233, 266, 4099, 4101, 4105, 5770]
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
S = {t: sched(t) for t in ("main", "peel")}
g = lambda t, p, m: d[f"{t}/{p}/{m}"]["ids"]
fd = lambda a, b: next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None) if a != b else None
def overlap(t, p):
    v = S[t].get(p, [])
    return len(v) >= 2 and v[0][1] == 0 and v[1][1] >= 1, v
ok1 = 0
print("GATE 1 (peel build): solo == behind, overlap confirmed in the trace")
for p in G1:
    a, b = g("peel", p, "solo"), g("peel", p, "behind")
    ov, v = overlap("peel", p)
    eq = a == b and len(a) == 256
    mo = g("main", p, "solo") == g("main", p, "behind")
    print(f"  {p}: solo==behind {'EQ' if eq else 'DIFF@%s' % fd(a, b)}  main solo==behind {'EQ' if mo else 'DIFF@%s' % fd(g('main', p, 'solo'), g('main', p, 'behind'))}  "
          f"peel schedules solo {v[0][2] if v else '?'} behind {v[1][2] if len(v) > 1 else '?'} (decoding {v[1][1] if len(v) > 1 else '?'})  overlap {'OK' if ov else 'NOT CONFIRMED'}")
    ok1 += eq and ov
print(f"GATE 1: {ok1} of {len(G1)}")
ok2 = 0
print("GATE 2 (peel == main, solo and behind, 256 ids)")
for p in G2:
    s_eq = g("peel", p, "solo") == g("main", p, "solo"); b_eq = g("peel", p, "behind") == g("main", p, "behind")
    ov, v = overlap("peel", p)
    ovm, _ = overlap("main", p)
    print(f"  {p}: solo {'EQ' if s_eq else 'DIFF@%s' % fd(g('peel', p, 'solo'), g('main', p, 'solo'))}  behind {'EQ' if b_eq else 'DIFF@%s' % fd(g('peel', p, 'behind'), g('main', p, 'behind'))}  overlap peel {'OK' if ov else 'NO'} main {'OK' if ovm else 'NO'}")
    ok2 += s_eq and b_eq
print(f"GATE 2: {ok2} of {len(G2)}")
