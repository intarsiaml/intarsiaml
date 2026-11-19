#!/usr/bin/env python3
"""cvatflip ABBA (ab.sh, several cells, REPS reps per server): per cell and server the median decode tok/s of its reps; per round the ratio
NUM/DEN; the median of the per-round ratios (headline), the mean and SE. Also: whether greedy replies (sha) are equal across arms, and the
per-server mean over all cells of tok/s relative to the other servers of the same arm (flags a server that is slow in every request).
usage: ab_an.py DIR[,DIR2,...] [NUM/DEN ...]   (default bk/base)   DIR holds <arm>-r<round>.jsonl; several dirs are pooled as further rounds (round numbers + 10 per dir; no log part)"""
import json, glob, re, sys, statistics as st, math, collections, gzip, os
DS = sys.argv[1].split(","); D = DS[0]; pairs = [p.split("/") for p in (sys.argv[2:] or ["bk/base"])]
tps = {}; shas = {}; cells = []
for k, d_ in enumerate(DS):
  for f in sorted(glob.glob(f"{d_}/*-r*.jsonl")):
    a, r = re.search(r"/([^/]+)-r(\d+)\.jsonl$", f).groups(); r = int(r) + 10 * k
    for x in (json.loads(l) for l in open(f) if l.startswith("{")):
        if x["cell"] not in cells: cells.append(x["cell"])
        tps.setdefault((a, int(r), x["cell"]), []).append(x["dec_tps"])
        shas.setdefault((a, int(r), x["cell"]), []).append(x["sha"])
arms = sorted({k[0] for k in tps}); rounds = sorted({k[1] for k in tps})
print("decode tok/s, median of the reps per server (reps in brackets):")
for c in cells:
    print(f" {c}")
    for a in arms:
        print(f"  {a:6s} " + "  ".join(f"r{r} {st.median(tps[(a, r, c)]):.2f} [{' '.join(f'{v:.1f}' for v in tps[(a, r, c)])}]" for r in rounds if (a, r, c) in tps))
for c in cells:
    for r in rounds:
        s = {tuple(shas[(a, r, c)]) for a in arms if (a, r, c) in shas}
        print(f"replies {c} r{r}: {'equal across arms' if len(s) == 1 else 'DIFFER'} ({len(shas[(arms[0], r, c)])} reps)") if s else None
print("ratios of per-round medians:")
allr = collections.defaultdict(list)
for n, d in pairs:
    for c in cells + ["ALL"]:
        x = []
        for r in rounds:
            if c == "ALL":
                v = [st.median(tps[(n, r, k)]) / st.median(tps[(d, r, k)]) for k in cells if (n, r, k) in tps and (d, r, k) in tps]
                if v: x.append(st.mean(v))
            elif (n, r, c) in tps and (d, r, c) in tps:
                x.append(st.median(tps[(n, r, c)]) / st.median(tps[(d, r, c)]))
        if x:
            se = st.stdev(x) / math.sqrt(len(x)) if len(x) > 1 else float("nan")
            print(f" {n}/{d} {c:11s} " + " ".join(f"{v:.4f}" for v in x) + f"  median {st.median(x):.4f}  mean {st.mean(x):.4f} (SE {se:.4f})  >1: {sum(v > 1 for v in x)}/{len(x)}")
print("per server, mean over cells of tok/s / the arm's own mean over rounds (a server slow in every request shows well below 1):")
for a in arms:
    m = {r: st.mean(st.median(tps[(a, r, c)]) / st.mean([st.median(tps[(a, q, c)]) for q in rounds if (a, q, c) in tps]) for c in cells if (a, r, c) in tps) for r in rounds}
    print(f"  {a:6s} " + "  ".join(f"r{r} {v:.4f}" for r, v in m.items()))

# serve logs (INTARSIA_SPEC_TRACE=1): per server and request the mean graph step (ms, positions >= 100) and the eager steps (a token line with no graph step line after it)
T = re.compile(r"^spec: token \d+ pos (\d+) drafted")
G = re.compile(r"^spec: mtp graph step ([\d.]+) ms gap [\d.]+ ms n (\d+) len (\d+)")
def parse(fn):
    op = gzip.open(fn, "rt", errors="replace") if fn.endswith(".gz") else open(fn, errors="replace")
    reqs = [{"g": [], "eager": 0}]; last = -1; pend = False
    for l in op:
        t = T.match(l)
        if t:
            if pend: reqs[-1]["eager"] += 1
            pos = int(t.group(1))
            if pos < last - 50: reqs.append({"g": [], "eager": 0})
            last = pos; pend = True; continue
        g = G.match(l)
        if g and pend:
            if int(g.group(3)) >= 100: reqs[-1]["g"].append(float(g.group(1)))
            pend = False
    if pend: reqs[-1]["eager"] += 1
    return reqs
logs = {}
for f in sorted(glob.glob(f"{D}/serve-*-r*.log*")):
    m = re.search(r"serve-(.+)-r(\d+)\.log(?:\.gz)?$", f)
    logs[(m.group(1), int(m.group(2)))] = parse(f)
if logs and len(DS) == 1:
    print("mean graph step (ms, positions >= 100) per cell, and eager steps (token lines without a graph step) per cell, per server; requests are mapped by order: 2 warm-ups then the cells x reps:")
    nreq = {k: len(v) for k, v in logs.items()}
    reps = max(len(v) for v in tps.values())
    for a in arms:
        for r in rounds:
            R = logs.get((a, r))
            if not R: continue
            allg = [x for q in R for x in q["g"]]
            row = f"  {a:6s} r{r}: all steps {st.mean(allg):.3f} ms, eager {sum(q['eager'] for q in R)}, requests {len(R)}"
            if len(R) == 2 + reps * len(cells):
                for i, c in enumerate(cells):
                    q = R[2 + i * reps: 2 + (i + 1) * reps]; g = [x for z in q for x in z["g"]]
                    row += f" | {c} {st.mean(g):.3f} e{sum(z['eager'] for z in q)}"
            print(row)
