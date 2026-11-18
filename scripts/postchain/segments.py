#!/usr/bin/env python3
"""postchain: n20 minus base per segment between chains (steps matched by request, position and n; drafts are identical there).
Prints, per round, each chain (request, position, accepted, head catch-up rows) and the mean step difference of the steps after it
up to the next chain. usage: segments.py DIR [--arm n20] [--base base]"""
import re, glob, gzip, argparse, statistics as st
ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--arm", default="n20"); ap.add_argument("--base", default="base")
a = ap.parse_args()
T = re.compile(r"^spec: token \d+ pos (\d+) drafted (\d+) accepted (\d+)")
G = re.compile(r"^spec: mtp graph step ([\d.]+) ms gap [\d.]+ ms n (\d+) len (\d+)")
C = re.compile(r"^spec: copy token \d+ pos (\d+) drafted (\d+) accepted (\d+)")
def opn(f): return gzip.open(f, "rt", errors="replace") if f.endswith(".gz") else open(f, errors="replace")
def parse(fn):
    ev, req, last, prev_acc = [], -1, -1, 0
    for l in opn(fn):
        t = T.match(l)
        if t: prev_acc = int(t.group(3)); continue
        c = C.match(l)
        if c: ev.append(("chain", req, int(c.group(1)), int(c.group(3)), prev_acc + 1 + int(c.group(3)))); continue
        g = G.match(l)
        if g:
            pos = int(g.group(3))
            if pos < last - 50 or req < 0: req += 1
            last = pos
            ev.append(("step", req, pos, int(g.group(2)), float(g.group(1))))
    return ev
for fb in sorted(glob.glob(f"{a.dir}/serve-{a.base}-r*.log*")):
    rnd = re.search(r"-r(\d+)\.log", fb).group(1)
    fa = glob.glob(f"{a.dir}/serve-{a.arm}-r{rnd}.log*")
    if not fa: continue
    B = {(e[1], e[2], e[3]): e[4] for e in parse(fb) if e[0] == "step"}
    segs, cur = [], ["start", []]
    for e in parse(fa[0]):
        if e[0] == "chain":
            segs.append(cur); cur = [f"chain req {e[1]} pos {e[2]} acc {e[3]} catch-up {e[4]}", []]
        elif (e[1], e[2], e[3]) in B and e[1] >= 2:
            cur[1].append(e[4] - B[(e[1], e[2], e[3])])
    segs.append(cur)
    print(f"r{rnd}: " + " | ".join(f"{lab}: {st.mean(d):+.3f} ({len(d)})" for lab, d in segs if d))
