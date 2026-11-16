#!/usr/bin/env python3
"""dv27: tables from speed-<visit>-<mode>-<set>[-nt]-T<temp>.json (sweep.sh). table.py <raw dir> [baseline arm, default V65536]
Visit names are <ARM>.<tag>-<mode> (V65536.a-s1). One block per (mode, set, thinking, temperature, streams); the baseline's reply hashes
are the identity reference (drafting must not change output)."""
import glob, json, statistics, sys
from collections import defaultdict

d = sys.argv[1]; base = sys.argv[2] if len(sys.argv) > 2 else "V65536"
cells = defaultdict(list)
for f in sorted(glob.glob(f"{d}/speed-*.json")):
    r = json.load(open(f)); name = r["arm"]; mode = name.rsplit("-", 1)[1]; arm = name.rsplit("-", 1)[0].split(".")[0]
    for row in r["rows"]:
        row["_visit"] = name
        cells[(mode, r["set"], r["thinking"], r["temperature"], row["streams"], arm)].append(row)
num = lambda a: (a != base, int(a[1:]) if a[1:].isdigit() else -1)
for key in sorted({k[:5] for k in cells}):
    arms = sorted({k[5] for k in cells if k[:5] == key}, key=num)
    if base not in arms:
        continue
    mode, st, th, T, n = key
    ref = cells[key + (base,)]
    bsha = [e["sha256"] if e else None for e in ref[0]["each"]]
    bm = statistics.mean(x["decode_tok_s_combined"]["median"] for x in ref)
    print(f"\n### {mode}: set {st}, thinking {th}, T {T:g}, {n} stream(s)\n")
    print(f"| arm | visits | combined tok/s per visit | mean | vs {base} | ITL p99 ms | accept | tok/pass | CJK share | identical to {base} |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for a in arms:
        rr = cells[key + (a,)]
        c = [x["decode_tok_s_combined"]["median"] for x in rr]; m = statistics.mean(c)
        prop = sum(x["mtp"]["drafted"] for x in rr); acc = sum(x["mtp"]["accepted"] for x in rr)
        itl = statistics.mean(x["itl_ms_p99"] for x in rr)
        tpp = [x["mtp"]["tokens_per_pass"] for x in rr if x["mtp"]["tokens_per_pass"]]
        same = sum(1 for x in rr for i, e in enumerate(x["each"]) if e and e["sha256"] == bsha[i]); tot = sum(len(x["each"]) for x in rr)
        print(f"| {a} | {len(rr)} | {' / '.join('%.2f' % v for v in c)} | {m:.2f} | {m / bm:.3f} | {itl:.0f} | {('%.3f' % (acc / prop)) if prop else '-'} | "
              f"{('%.2f' % statistics.mean(tpp)) if tpp else '-'} | {statistics.mean(x['cjk_share'] for x in rr):.2f} | {same} / {tot} |")
print("\n### pivot: mean combined tok/s relative to " + base + " (ratio; <1 loses)\n")
allarms = sorted({k[5] for k in cells}, key=num)
print("| mode | set | thinking | T | streams | " + " | ".join(allarms) + " |"); print("|---|---|---|---|---|" + "---|" * len(allarms))
for key in sorted({k[:5] for k in cells}):
    if key + (base,) not in cells:
        continue
    bm = statistics.mean(x["decode_tok_s_combined"]["median"] for x in cells[key + (base,)])
    vals = [("%.3f" % (statistics.mean(x["decode_tok_s_combined"]["median"] for x in cells[key + (a,)]) / bm)) if key + (a,) in cells else "-" for a in allarms]
    print(f"| {key[0]} | {key[1]} | {key[2]} | {key[3]:g} | {key[4]} | " + " | ".join(vals) + " |")
