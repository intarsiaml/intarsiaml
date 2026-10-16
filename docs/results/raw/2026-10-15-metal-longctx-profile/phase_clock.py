#!/usr/bin/env python3
"""GPU clock/power/temperature during the PREFILL window of each generate run (approximate: process end - decode time - 0.3 s
teardown, back by the prefill time) and during the DECODE window. usage: phase_clock.py"""
import json, os, re, glob, statistics as st, datetime
D = os.path.dirname(os.path.abspath(__file__))
def ts(s): return datetime.datetime.fromisoformat(s).timestamp()
mm = []
for l in open(f"{D}/clock/macmon.jsonl"):
    try:
        j = json.loads(l); mm.append((ts(j["timestamp"]), j["gpu_usage"][0], j["gpu_usage"][1], j["temp"]["gpu_temp_avg"], j["gpu_power"]))
    except Exception: pass
runs = {l.split("\t")[2].strip(): (float(l.split("\t")[0]), float(l.split("\t")[1])) for l in list(open(f"{D}/runs.tsv"))[1:]}
def win(a, b):
    w = [x for x in mm if a <= x[0] <= b]
    if len(w) < 3: return None
    f = sorted(x[1] for x in w)
    return dict(n=len(w), med=st.median(f), p10=f[len(f)//10], low=100*sum(1 for x in f if x < 1450)/len(f), pw=st.median(x[4] for x in w), t=max(x[3] for x in w), us=st.median(x[2] for x in w))
print("| run | prefill ms | prefill window: median MHz | p10 | % samples < 1450 | median W | median util | max C | decode window: median MHz | median W |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
rows = {}
for name, (a, b) in runs.items():
    p = None
    for d in ("plain", "cb", "kernel"):
        f = f"{D}/{d}/{name}.err"
        if os.path.exists(f): p = f
    if not p: continue
    s = open(p).read()
    m = re.search(r"prefill (\d+) ms \(([\d.]+) tok/s\) \| decode (\d+) ms", s)
    if not m: continue
    pf, dc = int(m.group(1)) / 1e3, int(m.group(3)) / 1e3
    pe = b - dc - 0.3
    w = win(pe - pf + 0.05 * pf, pe - 0.05 * pf)
    wd = win(pe + 0.2, b - 0.4) if dc > 1 else None
    if w:
        print(f"| {name} | {pf*1e3:.0f} | {w['med']:.0f} | {w['p10']:.0f} | {w['low']:.0f}% | {w['pw']:.0f} | {w['us']:.2f} | {w['t']:.0f} | " + (f"{wd['med']:.0f} | {wd['pw']:.0f} |" if wd else " | |"))
