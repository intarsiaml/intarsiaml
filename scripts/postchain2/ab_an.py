#!/usr/bin/env python3
"""postchain2 ABBA (ab.sh, code-gen 800 tokens, REPS reps per server): per server the median decode tok/s of its code-gen requests and the mean
graph step (ms); per round the ratios; mean and SE over rounds. usage: ab_an.py DIR [num/den ...] (default n20/base nx/bx nx/base bx/base)"""
import json, glob, re, sys, gzip, statistics as st, math
D = sys.argv[1]; pairs = [p.split("/") for p in (sys.argv[2:] or ["n20/base", "nx/bx", "nx/base", "bx/base"])]
G = re.compile(r"^spec: mtp graph step ([\d.]+) ms")
tps, step, shas = {}, {}, {}
for f in sorted(glob.glob(f"{D}/*-r*.jsonl")):
    a, r = re.search(r"/([^/]+)-r(\d+)\.jsonl$", f).groups()
    j = [json.loads(l) for l in open(f) if l.startswith("{")]
    j = [x for x in j if x["cell"] == "code-gen"]
    tps[(a, int(r))] = st.median(x["dec_tps"] for x in j)
    shas[(a, int(r))] = tuple(x["sha"] for x in j)
    lf = f"{D}/serve-{a}-r{r}.log"
    try:
        op = gzip.open(lf + ".gz", "rt") if not glob.glob(lf) else open(lf, errors="replace")
        v = [float(m.group(1)) for l in op if (m := G.match(l))]
        step[(a, int(r))] = st.mean(v) if v else float("nan")
    except FileNotFoundError:
        step[(a, int(r))] = float("nan")
arms = sorted({a for a, _ in tps}); rounds = sorted({r for _, r in tps})
print("decode tok/s, median of the code-gen reps (mean graph step ms):")
for a in arms:
    print(f"  {a:6s} " + "  ".join(f"r{r} {tps[(a, r)]:.2f} ({step[(a, r)]:.3f})" for r in rounds if (a, r) in tps))
for r in rounds:
    s = {shas[(a, r)] for a in arms if (a, r) in shas}
    print(f"round {r}: replies {'equal across arms' if len(s) == 1 else 'DIFFER'}")
for n, d in pairs:
    x = [tps[(n, r)] / tps[(d, r)] for r in rounds if (n, r) in tps and (d, r) in tps]
    if x:
        se = st.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
        print(f"{n}/{d}: " + " ".join(f"{v:.4f}" for v in x) + f"  mean {st.mean(x):.4f} (SE {se:.4f})")
