#!/usr/bin/env python3
"""Summaries of speed2.sh output (markdown). python3 speedsum2.py <speed2 dir>"""
import glob, json, re, sys, statistics as st, collections, datetime
D = sys.argv[1]
def iso(t): return datetime.datetime.fromisoformat(t).timestamp()
mac = []
for l in open(f"{D}/macmon.jsonl"):
    try: j = json.loads(l)
    except Exception: continue
    mac.append((iso(j["timestamp"]), j["gpu_usage"][0], j["gpu_usage"][1], j["gpu_power"]))
def clock(t0, t1):
    v = [m[1] for m in mac if t0 <= m[0] <= t1 and m[2] > 0.5]
    return st.median(v) if v else float("nan")
def fmt(v, d=1): return f"{st.median(v):.{d}f} [{min(v):.{d}f}..{max(v):.{d}f}]" if v else "-"
def stats(rs):
    busy = sum(r["e"] - r["s"] for r in rs) * 1000; span = (rs[-1]["e"] - rs[0]["s"]) * 1000
    return busy, span, span - busy, len(rs)
mk = {}
for l in open(f"{D}/markers.txt"):
    p = l.split()
    if len(p) == 6: mk[(p[0], int(p[1][1:]), int(p[2]), int(p[3][1:]))] = (float(p[4]), float(p[5]))
# uptime-clock (cb traces, cbgap-bench t0/t1) -> epoch (macmon, markers): the marker t1 of a cold run is taken right after its process
# exits, a few ms after its last command buffer ended; the median of that difference over all cold runs is the offset (good to ~10 ms)
_off = []
for (arm, r, size, c), (t0, t1) in mk.items():
    try: tr = [json.loads(l) for l in open(f"{D}/cold/{arm}-r{r}-{size}-c{c}.trace.jsonl")]
    except OSError: continue
    _off.append(t1 - tr[-1]["e"])
OFF = st.median(_off) if _off else 0.0
C = collections.defaultdict(list)  # (size, chunk, arm) -> dicts
for f in sorted(glob.glob(f"{D}/cold/*.trace.jsonl")):
    m = re.fullmatch(r".*/(\w+)-r(\d+)-(\d+)-c(\d+)\.trace\.jsonl", f)
    arm, r, size, c = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    tr = [json.loads(l) for l in open(f)]
    try: wall = float(re.search(r"prefill (\d+) ms", open(f.replace(".trace.jsonl", ".err")).read()).group(1))
    except Exception: continue
    b, s, i, n = stats(tr)
    t = mk.get((arm, r, size, c))
    C[(size, c, arm)].append(dict(r=r, wall=wall, busy=b, span=s, idle=i, cbs=n, clk=clock(*t) if t else float("nan")))
print("### cold: fresh `intarsia generate`, 8 rounds, position-balanced\n")
print("| tokens | chunk | arm | n | prefill wall ms | GPU busy ms | idle in span ms | cbs | GPU MHz |")
print("|---|---|---|---|---|---|---|---|---|")
for size in (1024, 4062):
    for c in (256, 2048):
        for a in ("base", "n1", "n1np", "n1ns"):
            v = C.get((size, c, a))
            if not v: continue
            col = lambda k: [x[k] for x in v]
            clk = [x for x in col("clk") if x == x]
            print(f"| {size} | {c} | {a} | {len(v)} | {fmt(col('wall'))} | {fmt(col('busy'))} | {fmt(col('idle'),2)} | {st.median(col('cbs')):.0f} | {st.median(clk) if clk else float('nan'):.0f} |")
def paired(size, c, num, den, key):
    a = {x["r"]: x[key] for x in C.get((size, c, num), [])}; b = {x["r"]: x[key] for x in C.get((size, c, den), [])}
    d = [a[r] - b[r] for r in a if r in b]; q = [a[r] / b[r] for r in a if r in b]
    return d, q
print("\n### cold: paired by round (same cell, same round), median [min..max]\n")
print("| tokens | chunk | n1 - base wall ms | n1 / base wall | n1np - base wall ms | n1ns - n1 wall ms | n1 - base busy ms |")
print("|---|---|---|---|---|---|---|")
for size in (1024, 4062):
    for c in (256, 2048):
        d1, q1 = paired(size, c, "n1", "base", "wall"); d2, _ = paired(size, c, "n1np", "base", "wall"); d3, _ = paired(size, c, "n1ns", "n1", "wall"); d4, _ = paired(size, c, "n1", "base", "busy")
        print(f"| {size} | {c} | {fmt(d1)} | {fmt(q1,3)} | {fmt(d2)} | {fmt(d3)} | {fmt(d4)} |")
print("\n### cold: c256 vs c2048 within an arm, paired by round (the extra cost of 256-token chunks)\n")
print("| tokens | arm | wall c256 / c2048 | wall c256 - c2048 ms | busy c256 / c2048 |")
print("|---|---|---|---|---|")
for size in (1024, 4062):
    for a in ("base", "n1"):
        x = {s["r"]: s for s in C.get((size, 256, a), [])}; y = {s["r"]: s for s in C.get((size, 2048, a), [])}
        rs = [r for r in x if r in y]
        print(f"| {size} | {a} | {fmt([x[r]['wall']/y[r]['wall'] for r in rs],3)} | {fmt([x[r]['wall']-y[r]['wall'] for r in rs])} | {fmt([x[r]['busy']/y[r]['busy'] for r in rs],3)} |")

# warm A/B
W = collections.defaultdict(list)
for f in sorted(glob.glob(f"{D}/warm/*.trace.jsonl")):
    m = re.fullmatch(r".*/c(\d+)-p(\d+)\.trace\.jsonl", f)
    c, p = int(m.group(1)), int(m.group(2))
    tr = [json.loads(l) for l in open(f)]
    for q in map(json.loads, open(f.replace(".trace.jsonl", ".req.jsonl"))):
        if q["mode"] not in ("on", "off"): continue
        rs = [r for r in tr if r["s"] >= q["t0"] - 1e-3 and r["e"] <= q["t1"] + 1e-3]
        if not rs: continue
        b, s, i, n = stats(rs)
        W[(q["len"], c)].append(dict(p=p, rep=q["rep"], mode=q["mode"], wall=q["prefill_ms"], busy=b, span=s, idle=i, cbs=n, clk=clock(q["t0"] + OFF, q["t1"] + OFF)))
print("\n### warm: staged uploads on vs off, alternating per request inside one process (ABBA), 3 processes x 12 reps\n")
print("| tokens | chunk | mode | n | prefill wall ms | GPU busy ms | idle in span ms | cbs | GPU MHz |")
print("|---|---|---|---|---|---|---|---|---|")
for (size, c), v in sorted(W.items()):
    for mode in ("off", "on"):
        u = [x for x in v if x["mode"] == mode]
        col = lambda k: [x[k] for x in u]
        print(f"| {size} | {c} | {mode} | {len(u)} | {fmt(col('wall'))} | {fmt(col('busy'))} | {fmt(col('idle'),2)} | {st.median(col('cbs')):.0f} | {st.median(col('clk')):.0f} |")
print("\n### warm: paired on / off (the two requests of one repetition, same process)\n")
print("| tokens | chunk | wall on / off | wall on - off ms | busy on / off | pairs |")
print("|---|---|---|---|---|---|")
for (size, c), v in sorted(W.items()):
    on = {(x["p"], x["rep"]): x for x in v if x["mode"] == "on"}; off = {(x["p"], x["rep"]): x for x in v if x["mode"] == "off"}
    ks = [k for k in on if k in off]
    print(f"| {size} | {c} | {fmt([on[k]['wall']/off[k]['wall'] for k in ks],3)} | {fmt([on[k]['wall']-off[k]['wall'] for k in ks])} | {fmt([on[k]['busy']/off[k]['busy'] for k in ks],3)} | {len(ks)} |")
