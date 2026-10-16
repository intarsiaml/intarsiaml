#!/usr/bin/env python3
"""Run table with GPU clock/temperature per run, plain-run prefill/decode medians, derived floors.
usage: summarize.py   (reads runs.tsv, clock/macmon.jsonl, plain/*.err in this directory)"""
import json, os, re, glob, statistics as st, datetime, collections
D = os.path.dirname(os.path.abspath(__file__))
def ts(s): return datetime.datetime.fromisoformat(s).timestamp()
mm = []
for l in open(f"{D}/clock/macmon.jsonl"):
    try:
        j = json.loads(l); mm.append((ts(j["timestamp"]), j["gpu_usage"][0], j["gpu_usage"][1], j["temp"]["gpu_temp_avg"], j["gpu_power"]))
    except Exception: pass
runs = [l.rstrip("\n").split("\t") for l in open(f"{D}/runs.tsv")][1:]
def clock(a, b):
    w = [x for x in mm if a <= x[0] <= b and x[2] > 0.5]
    if not w: return None
    f = sorted(x[1] for x in w)
    return dict(n=len(w), med=st.median(f), p10=f[len(f)//10], mn=f[0], low=sum(1 for x in f if x < 1450) / len(f), tmax=max(x[3] for x in w), pw=st.median(x[4] for x in w))
print("| run | wall s | busy samples | median MHz | p10 MHz | min MHz | share < 1450 | max GPU C | median W |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|")
clk = {}
for a, b, lab in runs:
    a, b = float(a), float(b); c = clock(a, b); clk[lab] = c
    if c: print(f"| {lab} | {b-a:.1f} | {c['n']} | {c['med']:.0f} | {c['p10']:.0f} | {c['mn']:.0f} | {100*c['low']:.0f}% | {c['tmax']:.0f} | {c['pw']:.0f} |")
    else: print(f"| {lab} | {b-a:.1f} | 0 | | | | | | |")
print()
# plain runs
def parse(p):
    s = open(p).read()
    m = re.search(r"prompt: (\d+) tokens", s); n = int(m.group(1)) if m else None
    m = re.search(r"prefill (\d+) ms \(([\d.]+) tok/s\) \| decode (\d+) ms \(([\d.]+) tok/s\)", s)
    return n, (int(m.group(1)), float(m.group(2)), int(m.group(3)), float(m.group(4))) if m else None
tab = collections.defaultdict(list)
for p in sorted(glob.glob(f"{D}/plain/ctx*-r*.err")):
    name = os.path.basename(p)[:-4]; n, r = parse(p)
    if r: tab[re.sub(r"-r\d+$", "", name)].append((name, n, r))
print("| config | prompt tok | n | prefill ms (median, min-max) | prefill tok/s | decode tok/s (median, min-max) | runs below 1450 MHz median |\n|---|---:|---:|---|---:|---|---|")
for k, v in tab.items():
    ok = [x for x in v if clk.get(x[0]) and clk[x[0]]["med"] >= 1450]
    pf = [x[2][0] for x in v]; dc = [x[2][3] for x in v if x[2][2] > 1]
    low = [x[0] for x in v if clk.get(x[0]) and clk[x[0]]["med"] < 1450]
    print(f"| {k} | {v[0][1]} | {len(v)} | {st.median(pf):.0f} ({min(pf)}-{max(pf)}) | {v[0][1]/st.median(pf)*1000:.0f} | " + (f"{st.median(dc):.2f} ({min(dc):.2f}-{max(dc):.2f})" if dc else "") + f" | {','.join(low) or '-'} |")
# floors
W = 15.1368e9; KVB = 36 * 8 * 128 * 2 * 2; BW = 614e9
print("\n| context | KV bytes/token | bytes read/token | floor ms | floor tok/s | weights-only floor tok/s |\n|---:|---:|---:|---:|---:|---:|")
for n in (128, 1024, 4096, 16384, 32768, 40000, 65536):
    b = W + KVB * n
    print(f"| {n} | {KVB*n/1e6:.0f} MB | {b/1e9:.3f} GB | {b/BW*1e3:.2f} | {BW/b:.2f} | {BW/W:.2f} |")
