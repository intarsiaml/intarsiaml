#!/usr/bin/env python3
"""postchain: are the MTP head's drafts the same in base and n20 at the same (request, position, n)? Before / after the first chain.
usage: drafts_id.py DIR [--arm n20] [--base base] [--warm 2]"""
import re, sys, glob, gzip, argparse, collections
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--arm", default="n20"); ap.add_argument("--base", default="base"); ap.add_argument("--warm", type=int, default=2)
a = ap.parse_args()
T = re.compile(r"^spec: token \d+ pos (\d+) drafted (\d+) accepted (\d+) emits \d+ drafts \[([^\]]*)\]")
G = re.compile(r"^spec: mtp graph step ([\d.]+) ms gap [\d.]+ ms n (\d+) len (\d+)")
C = re.compile(r"^spec: copy token")
def opn(f): return gzip.open(f, "rt", errors="replace") if f.endswith(".gz") else open(f, errors="replace")
def parse(fn):
    reqs, cur, last, pend, chains = [], None, -1, None, 0
    for l in opn(fn):
        t = T.match(l)
        if t: pend = (int(t.group(3)), t.group(4)); continue
        if C.match(l): chains += 1; pend = None; continue
        g = G.match(l)
        if g:
            pos = int(g.group(3))
            if cur is None or pos < last - 50: cur = {}; reqs.append(cur)
            last = pos
            if pend: cur[(pos, int(g.group(2)))] = (pend[0], pend[1], chains > 0, float(g.group(1)))
            pend = None
    return reqs[a.warm:]
for fb in sorted(glob.glob(f"{a.dir}/serve-{a.base}-r*.log*")):
    rnd = re.search(r"-r(\d+)\.log", fb).group(1)
    fa = glob.glob(f"{a.dir}/serve-{a.arm}-r{rnd}.log*")
    if not fa: continue
    A, B = parse(fa[0]), parse(fb)
    cnt = collections.Counter(); dms = collections.defaultdict(list)
    for ra, rb in zip(A, B):
        for k, (acc, dr, seen, ms) in ra.items():
            if k not in rb: continue
            same = (acc, dr) == rb[k][:2]
            cnt[(seen, same)] += 1
            dms[(seen, same)].append(ms - rb[k][3])
    print(f"r{rnd}: " + "  ".join(f"{'seen' if s else 'unseen'}/{'same' if e else 'DIFF'}: {cnt[(s,e)]} steps, ms diff {sum(dms[(s,e)])/max(1,len(dms[(s,e)])):+.3f}" for s in (False, True) for e in (True, False)))
