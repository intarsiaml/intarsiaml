#!/usr/bin/env python3
"""Join g0b_bench CSV rounds with a macmon GPU-clock trace, drop rounds whose chains ran under MIN_MHZ, and print
T(cand)/T(ref) tables (within-round ratios, medians, p10-p90).
usage: analyze_g0b.py <dir> <g0b.csv> [--min-mhz 1450]  (dir holds macmon.jsonl); writes <dir>/summary_<csvname>.json"""
import sys, os, json, csv, bisect, statistics, collections, datetime

d, csvp = sys.argv[1], sys.argv[2]
MIN_MHZ = float(sys.argv[sys.argv.index("--min-mhz") + 1]) if "--min-mhz" in sys.argv else 1450.0
mm = []
for line in open(os.path.join(d, "macmon.jsonl")):
    line = line.strip()
    if not line.startswith("{"):
        continue
    try:
        j = json.loads(line)
    except Exception:
        continue
    t = datetime.datetime.fromisoformat(j["timestamp"].replace("Z", "+00:00")).timestamp() * 1000
    mm.append((t, j["gpu_usage"][0]))
mm.sort()
mm_t = [x[0] for x in mm]

def freq_in(a, b):
    i, k = bisect.bisect_left(mm_t, a), bisect.bisect_right(mm_t, b)
    return [mm[x][1] for x in range(i, k)]

rows = list(csv.DictReader(open(csvp)))
groups = collections.defaultdict(list)
for r in rows:
    r["t"] = float(r["sec_per_dispatch"]); r["ep"] = int(r["epoch_ms"]); r["n"] = int(r["n_dispatch"])
    groups[(r["bench"], r["ctx"], r["round"], r["cand"].rstrip("0123456789") if r["bench"] == "small" else "")].append(r)
valid, clock = {}, {}
for key, rs in groups.items():
    a = min(x["ep"] - x["t"] * x["n"] * 1000 for x in rs) - 20
    b = max(x["ep"] for x in rs)
    fs = freq_in(a - 100, b + 100)
    valid[key] = bool(fs) and min(fs) >= MIN_MHZ
    clock[key] = (min(fs) if fs else None, statistics.mean(fs) if fs else None)

def ref_of(bench, cand, k):
    if bench == "attn":
        return "serving1"
    return cand.rstrip("0123456789") + "1"

by = collections.defaultdict(lambda: collections.defaultdict(list))
tref = {}
for key, rs in groups.items():
    for x in rs:
        if x["cand"].startswith("seq"):
            x["t"] *= int(x["k"])  # k back-to-back 1-row launches per verify step
for key, rs in groups.items():
    for x in rs:
        if x["cand"] == ref_of(x["bench"], x["cand"], x["k"]):
            tref[key] = x["t"]
nr = collections.Counter(); nv = collections.Counter()
for key in groups:
    nr[(key[0], key[1])] += 1; nv[(key[0], key[1])] += valid[key]
for key, rs in groups.items():
    if not valid[key] or key not in tref:
        continue
    for x in rs:
        by[(x["bench"], x["ctx"])][x["cand"]].append((x["t"], x["t"] / tref[key], int(x["k"])))

def q(v, p):
    v = sorted(v); r = p * (len(v) - 1); lo, hi = int(r), min(int(r) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (r - lo)

summ = {}
out = []
out.append(f"rounds with min GPU clock >= {MIN_MHZ:.0f} MHz (macmon 100 ms); ratios are within-round medians [p10-p90]\n")
out.append("| bench | ctx | rounds valid/total | mean clock of valid rounds (MHz) |\n|---|---|---|---|")
for bc in sorted(nr):
    cl = [clock[k][1] for k in groups if (k[0], k[1]) == bc and valid[k]]
    out.append(f"| {bc[0]} | {bc[1]} | {nv[bc]}/{nr[bc]} | {statistics.mean(cl) if cl else float('nan'):.0f} |")
out.append("")
for bc in sorted(by, key=lambda b: (b[0], int(b[1]))):
    out.append(f"### {bc[0]} ctx {bc[1]}\n")
    out.append("| cand | k | n | T (us) | T/T(ref) median | p10 | p90 |\n|---|---|---|---|---|---|---|")
    for cand, v in sorted(by[bc].items(), key=lambda kv: (kv[1][0][2], kv[0])):
        ts = [x[0] for x in v]; rs_ = [x[1] for x in v]
        s = dict(n=len(v), t_us=statistics.median(ts) * 1e6, ratio=statistics.median(rs_), p10=q(rs_, .1), p90=q(rs_, .9), k=v[0][2])
        summ[f"{bc[0]}/{bc[1]}/{cand}"] = s
        out.append(f"| {cand} | {s['k']} | {s['n']} | {s['t_us']:.2f} | {s['ratio']:.3f} | {s['p10']:.3f} | {s['p90']:.3f} |")
    out.append("")
name = os.path.splitext(os.path.basename(csvp))[0]
json.dump(summ, open(os.path.join(d, f"summary_{name}.json"), "w"), indent=1)
open(os.path.join(d, f"analysis_{name}.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out))
