#!/usr/bin/env python3
"""Lane dsv41/nccl-prefill: tables for docs/results/2026-11-05-dsv41-nccl.md from a run-full.sh window.

Usage: nccl-analyze.py RAW_DIR

RAW_DIR is run-full.sh's $OUT: ncclbench-r0-<tag>.txt (the 27-arm grid), ncclbench-dbg-r0-<tag>.txt (NCCL_DEBUG=INFO runs),
ncclbench-cold<lead>-r0-<tag>.txt (idle-start test), decision.txt / sessions.txt (nccl-prehook.sh), ab<N>/ (one directory per session:
steps.log, le-speed-*.log, le-quality-*.log, tok-intarsia-tp2.st), mem.log (both Sparks' MemAvailable every 5 s) and steps.log.
Prints Markdown to stdout.
"""
import glob
import json
import os
import re
import statistics as st
import sys
from collections import defaultdict

raw = sys.argv[1]
BASE = "pdefault-c2-b1M"


def table(path):
    t = {}
    for l in open(path, errors="replace"):
        m = re.match(r"rank 0\s+(\d+)\s+(.*)", l)
        if m:
            n = [float(x) for x in m.group(2).split()]
            if len(n) >= 7:
                t[int(m.group(1))] = n
    return t


def grid():
    arms = {}
    for f in sorted(glob.glob(f"{raw}/ncclbench-r0-*.txt")):
        arms[os.path.basename(f)[len("ncclbench-r0-"):-4]] = table(f)
    if not arms:
        return
    print("### A. all-reduce grid (nccl_bench, rank 0; ar_inplace)\n")
    print("| arm | 42 MB us | 42 MB GB/s | x base | 21 MB GB/s | 1 MiB us | 16 KiB us (pipe/sync) | 128 KiB us (pipe/sync) |")
    print("|---|---|---|---|---|---|---|---|")
    b = arms.get(BASE, {}).get(41943040)
    for tag, t in arms.items():
        r = t.get(41943040)
        if not r:
            print(f"| {tag} | no result | | | | | | |")
            continue
        g = r[-1]
        ratio = f"{g / b[-1]:.2f}" if b else "-"
        g21 = f"{t[21176320][-1]:.2f}" if 21176320 in t else "-"
        us1m = f"{t[1048576][0]:.0f}" if 1048576 in t else "-"
        s16 = f"{t[16384][0]:.0f}/{t[16384][1]:.0f}" if 16384 in t else "-"
        s128 = f"{t[131072][0]:.0f}/{t[131072][1]:.0f}" if 131072 in t else "-"
        print(f"| {tag} | {r[0]:.0f} | {g:.2f} | {ratio} | {g21} | {us1m} | {s16} | {s128} |")
    print()


def debug():
    fs = sorted(glob.glob(f"{raw}/ncclbench-dbg-r0-*.txt"))
    if not fs:
        return
    print("### A. what NCCL chose (NCCL_DEBUG=INFO, 42 MB runs)\n")
    for f in fs:
        tag = os.path.basename(f)[len("ncclbench-dbg-r0-"):-4]
        lines = open(f, errors="replace").read().splitlines()
        keep = [l for l in lines if re.search(r"Bytes -> Algo|nChannels|channels|Using|NCCL version|Connected all|proto|PROTO", l)]
        uniq = []
        for l in keep:
            l2 = re.sub(r"^\S+:\d+:\d+ \[\d+\] ", "", l)
            l2 = re.sub(r"0x[0-9a-f]+", "0x..", l2)
            if l2 not in uniq:
                uniq.append(l2)
        print(f"- `{tag}`: " + "; ".join(f"`{u[:140]}`" for u in uniq[:6]))
    print()


def cold():
    fs = sorted(glob.glob(f"{raw}/ncclbench-cold*-r0-*.txt"))
    if not fs:
        return
    print("### B. idle-start test (nccl_bench NB_COLD, median us)\n")
    for f in fs:
        for l in open(f, errors="replace"):
            if "cold test" in l or "cold all" in l or "lead all" in l:
                print("    " + l.strip())
    print()


def sessions():
    names = {}
    p = f"{raw}/sessions.txt"
    if os.path.exists(p):
        for l in open(p):
            a = l.split(None, 2)
            if len(a) >= 2:
                names[a[0]] = (a[1], a[2].strip() if len(a) > 2 else "")
    abs_ = sorted(glob.glob(f"{raw}/ab[0-9]*"), key=lambda p: int(re.sub(r"\D", "", os.path.basename(p))))
    if not abs_:
        return
    speed = {}
    cat = {}
    for d in abs_:
        ab = os.path.basename(d)
        rows = {}
        for f in glob.glob(f"{d}/le-speed-*.log"):
            for l in open(f, errors="replace"):
                m = re.match(r"\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*(\d+) \((\d+)[–-](\d+)\)\s*\|\s*(\d+)?", l)
                if m:
                    rows[int(m[1])] = (float(m[2]), int(m[3]), int(m[4]), int(m[5]))
        speed[ab] = rows
        for f in glob.glob(f"{d}/le-quality-*.log"):
            m = re.search(r"saved (\S+/result\.json)", open(f, errors="replace").read())
            if m and os.path.exists(m[1]):
                cat[ab] = m[1]
    print("### Sessions\n")
    print("| session | arm | env | prefill 1K tok/s (TTFT s) | 4K | 16K |")
    print("|---|---|---|---|---|---|")
    for d in abs_:
        ab = os.path.basename(d)
        nm, env = names.get(ab, ("?", ""))
        cells = []
        for want in (1024, 4096, 16384):
            hit = [v for k, v in speed.get(ab, {}).items() if abs(k - want) < want * 0.1]
            cells.append(f"{hit[0][1]} ({hit[0][2]}-{hit[0][3]}; {hit[0][0]:.2f} s)" if hit else "-")
        print(f"| {ab} | {nm} | {env or 'current'} | " + " | ".join(cells) + " |")
    print()
    # decode by kind (catspeed.py)
    kinds = defaultdict(dict)
    for ab, rd in cat.items():
        try:
            r = json.load(open(rd))
        except Exception:
            continue
        out = defaultdict(list)
        for ds in r["datasets"]:
            for c in ds["cases"]:
                s = c.get("speed") or {}
                tps = s.get("decode_tps") or s.get("decode_tok_s")
                n = s.get("completion_tokens")
                if tps and n:
                    out[c["name"].split("-")[0]].append((n, tps))
        allr = [x for v in out.values() for x in v]
        for k, v in list(out.items()) + [("all", allr)]:
            kinds[k][ab] = sum(n for n, _ in v) / sum(n / t for n, t in v)
    if kinds:
        print("### Decode by kind (tok/s, tokens over decode time; `catspeed.py` method)\n")
        cols = [os.path.basename(d) for d in abs_ if os.path.basename(d) in cat]
        print("| kind | " + " | ".join(f"{c} {names.get(c, ('?',))[0]}" for c in cols) + " |")
        print("|---|" + "---|" * len(cols))
        for k in sorted(kinds, key=lambda x: (x == "all", x)):
            print(f"| {k} | " + " | ".join(f"{kinds[k][c]:.1f}" if c in kinds[k] else "-" for c in cols) + " |")
        print()


def memory():
    p = f"{raw}/mem.log"
    if not os.path.exists(p):
        return
    lo = {}
    for l in open(p, errors="replace"):
        m = re.match(r"(\d\d:\d\d:\d\d) spark1=([\d.]+) spark2=([\d.]+)", l)
        if m:
            lo.setdefault("all", []).append((m[1], float(m[2]), float(m[3])))
    steps = open(f"{raw}/steps.log", errors="replace").read().splitlines() if os.path.exists(f"{raw}/steps.log") else []
    marks = []
    for l in steps:
        m = re.match(r"(\d\d:\d\d:\d\d) config (ab\d+)(?: done|:)", l)
        if m:
            marks.append((m[1], m[2], "done" in l))
    print("### Memory (MemAvailable GiB, lowest per session; kill line 5)\n")
    print("| session | spark1 | spark2 |")
    print("|---|---|---|")
    rows = lo.get("all", [])
    starts = {}
    for t, ab, done in marks:
        if not done:
            starts[ab] = t
        elif ab in starts:
            seg = [(a, b) for tt, a, b in rows if starts[ab] <= tt <= t]
            if seg:
                print(f"| {ab} | {min(a for a, _ in seg):.1f} | {min(b for _, b in seg):.1f} |")
    if rows:
        print(f"| whole window | {min(a for _, a, _ in rows):.1f} | {min(b for _, _, b in rows):.1f} |")
    print()


grid()
debug()
cold()
sessions()
memory()
