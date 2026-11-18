#!/usr/bin/env python3
"""postchain: how much of the n20 deficit is the post-chain step offset? Per round: total graph-step ms of the measured requests in each arm,
the summed offset of the steps matched by (request, position, n), chain pass ms, and the n20 / base decode-time ratio with the matched
offset removed. usage: ceiling.py DIR"""
import re, glob, gzip, sys, statistics as st
G = re.compile(r"^spec: mtp graph step ([\d.]+) ms gap [\d.]+ ms n (\d+) len (\d+)")
P = re.compile(r"^spec: copy pass \d+ rows ([\d.]+) ms")
def opn(f): return gzip.open(f, "rt", errors="replace") if f.endswith(".gz") else open(f, errors="replace")
def parse(fn, warm=2):
    steps, chains, req, last = {}, 0.0, -1, -1
    tot = 0.0
    for l in opn(fn):
        g = G.match(l)
        if g:
            pos = int(g.group(3))
            if pos < last - 50 or req < 0: req += 1
            last = pos
            if req >= warm:
                steps[(req, pos, int(g.group(2)))] = float(g.group(1)); tot += float(g.group(1))
            continue
        p = P.match(l)
        if p and req >= warm: chains += float(p.group(1))
    return steps, tot, chains
d = sys.argv[1]
for fb in sorted(glob.glob(f"{d}/serve-base-r*.log*")):
    rnd = re.search(r"-r(\d+)\.log", fb).group(1)
    fa = glob.glob(f"{d}/serve-n20-r{rnd}.log*")
    if not fa: continue
    B, tb, _ = parse(fb); A, ta, ca = parse(fa[0])
    off = sum(A[k] - B[k] for k in A if k in B)
    print(f"r{rnd}: base steps {tb/1e3:.2f} s, n20 steps {ta/1e3:.2f} s + chain passes {ca/1e3:.3f} s; matched offset {off/1e3:+.3f} s; "
          f"ratio base/n20 {tb/(ta+ca):.4f}, without the offset {tb/(ta+ca-off):.4f}")
