#!/usr/bin/env python3
"""Summaries of speed.sh output (markdown on stdout). python3 speedsum.py <speed dir>
cold: one `generate` process per sample; warm: requests of a cbgap-bench process after its untimed warmup, serving path.
busy = sum of command-buffer GPU times, span = first start to last end, idle = span - busy; clock = median GPU MHz over
macmon samples with the GPU busy (> 50 %) during that process. Paired ratios compare arms within the same round."""
import glob, json, re, statistics as st, sys, collections
D = sys.argv[1]
ARMS = ["base", "n1", "n1np", "n1ns"]

def clock(path):
    fr = []
    try:
        for l in open(path):
            try: j = json.loads(l)
            except Exception: continue
            if j["gpu_usage"][1] > 0.5: fr.append(j["gpu_usage"][0])
    except OSError: pass
    return st.median(fr) if fr else float("nan")

def fmt(v, d=1): return f"{st.median(v):.{d}f} [{min(v):.{d}f}..{max(v):.{d}f}]" if v else "-"

def cut(trace, t0, t1):
    return [r for r in trace if r["s"] >= t0 - 1e-3 and r["e"] <= t1 + 1e-3]

def stats(rs):
    busy = sum(r["e"] - r["s"] for r in rs) * 1000
    span = (rs[-1]["e"] - rs[0]["s"]) * 1000
    return busy, span, span - busy, len(rs)

# samples[(regime, size, chunk, arm)] = list of dict(round, wall, busy, span, idle, cbs, clk)
S = collections.defaultdict(list)
for f in sorted(glob.glob(f"{D}/cold/*.trace.jsonl")):
    m = re.fullmatch(r".*/(\w+)-r(\d+)-(\d+)-c(\d+)\.trace\.jsonl", f)
    arm, r, size, c = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
    tr = [json.loads(l) for l in open(f)]
    try: wall = float(re.search(r"prefill (\d+) ms", open(f.replace(".trace.jsonl", ".err")).read()).group(1))
    except Exception: continue
    b, s, i, n = stats(tr)
    S[("cold", size, c, arm)].append(dict(round=r, wall=wall, busy=b, span=s, idle=i, cbs=n, clk=clock(f.replace(".trace.jsonl", ".macmon.jsonl"))))
for f in sorted(glob.glob(f"{D}/warm/*.trace.jsonl")):
    m = re.fullmatch(r".*/(\w+)-r(\d+)-c(\d+)\.trace\.jsonl", f)
    arm, r, c = m.group(1), int(m.group(2)), int(m.group(3))
    tr = [json.loads(l) for l in open(f)]
    clk = clock(f.replace(".trace.jsonl", ".macmon.jsonl"))
    for q in map(json.loads, open(f.replace(".trace.jsonl", ".req.jsonl"))):
        if q["rep"] < 0: continue
        rs = cut(tr, q["t0"], q["t1"])
        if not rs: continue
        b, s, i, n = stats(rs)
        S[("warm", q["len"], c, arm)].append(dict(round=r, wall=q["prefill_ms"], busy=b, span=s, idle=i, cbs=n, clk=clk))

def table(regime):
    print(f"\n### {regime}\n")
    print("| tokens | chunk | arm | n | prefill wall ms | GPU busy ms | span ms | idle in span ms | cbs | GPU MHz |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for size in sorted({k[1] for k in S if k[0] == regime}):
        for c in (256, 2048):
            for a in ARMS:
                v = S.get((regime, size, c, a))
                if not v: continue
                col = lambda k: [x[k] for x in v]
                print(f"| {size} | {c} | {a} | {len(v)} | {fmt(col('wall'))} | {fmt(col('busy'))} | {fmt(col('span'))} | {fmt(col('idle'),2)} | {st.median(col('cbs')):.0f} | {st.median(col('clk')):.0f} |")

def paired(regime, key, num, den, size, c):
    """median over rounds of median(num)/median(den) of the same round."""
    a, b = collections.defaultdict(list), collections.defaultdict(list)
    for x in S.get((regime, size, c, num), []): a[x["round"]].append(x[key])
    for x in S.get((regime, size, c, den), []): b[x["round"]].append(x[key])
    rs = [st.median(a[r]) / st.median(b[r]) for r in a if r in b]
    return (st.median(rs), min(rs), max(rs), len(rs)) if rs else None

def ratios(regime):
    print(f"\n### {regime}: paired ratios (same round), median [min..max] over rounds\n")
    print("| tokens | chunk | n1 / base wall | n1 / base busy | n1 / base idle (ms diff) | n1np / base wall | n1ns / base wall |")
    print("|---|---|---|---|---|---|---|")
    for size in sorted({k[1] for k in S if k[0] == regime}):
        for c in (256, 2048):
            cell = lambda key, num, den: (lambda r: f"{r[0]:.3f} [{r[1]:.3f}..{r[2]:.3f}] n={r[3]}" if r else "-")(paired(regime, key, num, den, size, c))
            def dmed():
                a = [x["idle"] for x in S.get((regime, size, c, "n1"), [])]; b = [x["idle"] for x in S.get((regime, size, c, "base"), [])]
                return f"{st.median(a) - st.median(b):+.1f}" if a and b else "-"
            print(f"| {size} | {c} | {cell('wall','n1','base')} | {cell('busy','n1','base')} | {dmed()} | {cell('wall','n1np','base')} | {cell('wall','n1ns','base')} |")
    print(f"\n{regime}: c256 / c2048 within an arm (the extra cost of 256-token chunks), paired by round\n")
    print("| tokens | arm | wall | busy | span |")
    print("|---|---|---|---|---|")
    for size in sorted({k[1] for k in S if k[0] == regime}):
        for a in ARMS:
            def one(key):
                x, y = collections.defaultdict(list), collections.defaultdict(list)
                for s in S.get((regime, size, 256, a), []): x[s["round"]].append(s[key])
                for s in S.get((regime, size, 2048, a), []): y[s["round"]].append(s[key])
                rs = [st.median(x[r]) / st.median(y[r]) for r in x if r in y]
                return f"{st.median(rs):.3f} [{min(rs):.3f}..{max(rs):.3f}]" if rs else "-"
            print(f"| {size} | {a} | {one('wall')} | {one('busy')} | {one('span')} |")

for regime in ("cold", "warm"):
    table(regime)
    ratios(regime)
