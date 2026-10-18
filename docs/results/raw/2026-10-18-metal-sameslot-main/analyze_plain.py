import re, json, glob, statistics as st, datetime as dt
D = "plain"
def q(xs):
    xs = sorted(xs); n = len(xs)
    # inclusive-method quartiles
    def pct(p):
        k = (n - 1) * p; f = int(k); c = min(f + 1, n - 1); return xs[f] + (xs[c] - xs[f]) * (k - f)
    return pct(.25), pct(.75)
runs = {}
for f in sorted(glob.glob(f"{D}/*.err")):
    lab = f.split("/")[-1][:-4]
    t = open(f).read(); m = re.findall(r"prefill (\d+) ms \(([\d.]+) tok/s\)", t)[-1]
    idle = [int(x) for x in re.findall(r"=(\d+)", open(f"{D}/{lab}.idle").read())]
    rows = [json.loads(x) for x in open(f"{D}/{lab}.macmon.jsonl") if x.strip()]
    act = [r for r in rows if r["gpu_usage"][1] > 0.5]  # samples with GPU busy = the prefill window
    runs[lab] = dict(ms=int(m[0]), tps=float(m[1]), idle=st.median(idle), mhz=st.mean(r["gpu_usage"][0] for r in act) if act else float("nan"),
                     mhz_med=st.median(r["gpu_usage"][0] for r in act) if act else float("nan"), temp=max(r["temp"]["gpu_temp_avg"] for r in rows),
                     nact=len(act), power=st.mean(r["gpu_power"] for r in act) if act else float("nan"))
rounds = []
labs = sorted(l for l in runs if not l.startswith("warm"))
cand = [l for l in labs if l.startswith("cand")]; main = [l for l in labs if l.startswith("main")]
print("| run | order | prefill ms | tok/s | GPU idle (median of 3) | GPU MHz (macmon sample over the run) | macmon samples | GPU power W | GPU temp C |\n|---|---|---|---|---|---|---|---|---|")
for l in labs:
    r = runs[l]; print(f"| {l} | {int(l.split('-')[1])} | {r['ms']} | {r['tps']:.1f} | {r['idle']:.0f} | {r['mhz']:.0f} | {r['nact']} | {r['power']:.0f} | {r['temp']:.0f} |")
for l in ("warm-cand", "warm-main"):
    r = runs[l]; print(f"(discarded {l}: {r['ms']} ms, {r['tps']:.1f} tok/s)")
print()
for nm, g in (("CAND", cand), ("main", main)):
    ms = [runs[l]["ms"] for l in g]; tk = [runs[l]["tps"] for l in g]
    a, b = q(ms); c, d = q(tk)
    print(f"{nm}: n={len(g)} prefill ms median {st.median(ms):.1f} IQR {a:.1f}-{b:.1f} (min {min(ms)} max {max(ms)}); tok/s median {st.median(tk):.1f} IQR {c:.1f}-{d:.1f}")
# rounds: pair r = runs 2r-1, 2r
diffs = []; dm = []; dt_ = []; mainslower = 0
print("\n| round | order | CAND ms | main ms | main - CAND ms | main vs CAND % (ms) | CAND MHz | main MHz | MHz diff (main-CAND) | CAND temp | main temp |\n|---|---|---|---|---|---|---|---|---|---|---|")
for r in range(1, 9):
    ids = (2 * r - 1, 2 * r)
    pair = {}
    for l in labs:
        if int(l.split("-")[1]) in ids: pair[l.split("-")[0]] = l
    c, m = runs[pair["cand"]], runs[pair["main"]]
    pct = 100 * (m["ms"] / c["ms"] - 1); diffs.append(pct); mainslower += m["ms"] > c["ms"]
    dm.append(m["mhz"] - c["mhz"]); dt_.append(m["temp"] - c["temp"])
    order = "C M" if ids[0] == int(pair["cand"].split("-")[1]) else "M C"
    print(f"| {r} | {order} | {c['ms']} | {m['ms']} | {m['ms']-c['ms']:+d} | {pct:+.2f} % | {c['mhz']:.0f} | {m['mhz']:.0f} | {m['mhz']-c['mhz']:+.0f} | {c['temp']:.0f} | {m['temp']:.0f} |")
ties = sum(1 for r in range(1, 9) if False)
print(f"\npaired per-round difference main vs CAND (prefill ms): median {st.median(diffs):+.2f} %, mean {st.mean(diffs):+.2f} %, range {min(diffs):+.2f} to {max(diffs):+.2f} %; main slower in {mainslower}/8 rounds, faster in {sum(d<0 for d in diffs)}/8, equal {sum(d==0 for d in diffs)}/8")
print(f"medians: main/CAND ms {st.median([runs[l]['ms'] for l in main])/st.median([runs[l]['ms'] for l in cand]):.4f}")
def pearson(x, y):
    mx, my = st.mean(x), st.mean(y); sx = sum((a - mx) ** 2 for a in x) ** .5; sy = sum((b - my) ** 2 for b in y) ** .5
    return float('nan') if sx * sy == 0 else sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)
print(f"paired diff (%) vs paired clock diff (MHz): r = {pearson(diffs, dm):+.2f} (nan = clock identical in every run); vs paired temp diff: r = {pearson(diffs, dt_):+.2f} (n=8)")
allms = [runs[l]["ms"] for l in labs]; allmhz = [runs[l]["mhz"] for l in labs]; allt = [runs[l]["temp"] for l in labs]; allp = [runs[l]["power"] for l in labs]
print(f"all 16 runs: prefill ms vs clock r = {pearson(allms, allmhz):+.2f}; vs GPU temp r = {pearson(allms, allt):+.2f}; vs GPU power r = {pearson(allms, allp):+.2f}; clock range {min(allmhz):.0f}-{max(allmhz):.0f} MHz, temp range {min(allt):.0f}-{max(allt):.0f} C")
for nm, g in (("CAND", cand), ("main", main)):
    print(f"{nm} mean clock {st.mean(runs[l]['mhz'] for l in g):.0f} MHz, mean temp {st.mean(runs[l]['temp'] for l in g):.1f} C, median idle {st.median(runs[l]['idle'] for l in g):.0f} %")
