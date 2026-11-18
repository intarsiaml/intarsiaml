#!/usr/bin/env python3
"""mgdnv27: table of batch.sh output. python3 btable.py <dir with off-r<N>.jsonl / on-r<N>.jsonl>
Per round: aggregate tok/s of arm off and on (median of the 2 waves), ratio on/off; median of the round ratios; whether every request's reply sha (per wave, request index) is equal between the arms."""
import glob, json, os, re, statistics, sys

d = sys.argv[1]
agg, shas = {}, {}
for f in sorted(glob.glob(os.path.join(d, "*-r*.jsonl"))):
    arm, rnd = re.match(r"(.+)-r(\d+)\.jsonl", os.path.basename(f)).groups()
    for line in open(f):
        if line.startswith("AGG "):
            a = json.loads(line[4:])
            agg.setdefault((arm, int(rnd)), []).append(a["agg_tps"])
        elif line.startswith("{"):
            r = json.loads(line)
            shas.setdefault((int(rnd), r["i"]), {}).setdefault(arm, []).append(r["sha"])
rounds = sorted({k[1] for k in agg})
ratios = []
for r in rounds:
    o, n = agg.get(("off", r)), agg.get(("on", r))
    if o and n:
        ratios.append(statistics.median(n) / statistics.median(o))
        print(f"round {r}: off {o} on {n} -> {ratios[-1]:.3f}x")
if ratios:
    print(f"median of round ratios {statistics.median(ratios):.3f}x  (rounds {', '.join(f'{x:.3f}' for x in ratios)})")
    allo = [x for (a, _), v in agg.items() if a == "off" for x in v]
    alln = [x for (a, _), v in agg.items() if a == "on" for x in v]
    print(f"aggregate tok/s median off {statistics.median(allo):.1f} [{min(allo):.1f}-{max(allo):.1f}] on {statistics.median(alln):.1f} [{min(alln):.1f}-{max(alln):.1f}]")
eq = sum(1 for v in shas.values() if v.get("off") and v["off"] == v.get("on"))
tot = sum(1 for v in shas.values() if v.get("off") and v.get("on"))
print(f"reply shas equal between arms: {eq} of {tot} (round, request) pairs; unequal: {[k for k, v in shas.items() if v.get('off') != v.get('on')]}")
