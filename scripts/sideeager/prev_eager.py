#!/usr/bin/env python3
"""sideeager: per request of an ab.sh server (2 warm-ups + the cell's reps, one cell), tok/s by whether the PREVIOUS request ended on an eager step (a token line with no graph
step line after it, at the end of the request). usage: prev_eager.py DIR   (DIR from ab.sh with one cell; arms base / bk; pooled over rounds)"""
import re, sys, json, glob, statistics as st
T = re.compile(r"^spec: token \d+ pos (\d+) drafted"); G = re.compile(r"^spec: mtp graph step")
def ends(fn):
    reqs = [[]]; last = -1; pend = False
    for l in open(fn, errors="replace"):
        t = T.match(l)
        if t:
            if pend: reqs[-1].append("E")
            pos = int(t.group(1))
            if pos < last - 50: reqs.append([])
            last = pos; pend = True; continue
        if G.match(l) and pend: reqs[-1].append("g"); pend = False
    if pend: reqs[-1].append("E")
    return [r[-1] == "E" if r else False for r in reqs]
D = sys.argv[1]; res = {}
for f in sorted(glob.glob(f"{D}/serve-*-r*.log")):
    a, r = re.search(r"serve-(.+)-r(\d+)\.log", f).groups()
    e = ends(f); tps = [json.loads(l)["dec_tps"] for l in open(f"{D}/{a}-r{r}.jsonl") if l.startswith("{")]
    if len(e) != 2 + len(tps): print(f"{f}: {len(e)} requests in the log, {len(tps)} reps: skipped"); continue
    for i, v in enumerate(tps):
        res.setdefault((a, bool(e[1 + i])), []).append(v)   # request i of the cell is log request 2 + i; its predecessor is 1 + i
for k in sorted(res): print(f"{k[0]:5s} previous request ended on an eager step: {str(k[1]):5s} n {len(res[k]):3d}  mean tok/s {st.mean(res[k]):.2f}  median {st.median(res[k]):.2f}")
