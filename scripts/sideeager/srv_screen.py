#!/usr/bin/env python3
"""sideeager: screen servers for a persistent step-time offset (the bx r3 / bk r1 kind: every request ~0.2-0.3 ms slower from the first, no eager side-stream pass).
Servers whose requests are the same (greedy, same prompts) have identical step sequences, so step i of one server is matched to step i of the others.
  pair mode:  srv_screen.py pair DIR   DIR holds serve-base-rN.log / serve-bk-rN.log (ab.sh): per round, bk - base per 100-step window and overall
  group mode: srv_screen.py group LOG [LOG ...]   all logs from the same prompts (nsys2.sh servers): per server, its mean offset from the per-step median of the group and the
              per 100-step window profile (a persistent mode shows as a run of high windows)"""
import re, sys, glob, statistics as st
G = re.compile(r"^spec: mtp graph step ([\d.]+) ms gap ([\d.]+) ms n (\d+) len (\d+)")
T = re.compile(r"^spec: token \d+ pos (\d+) drafted (\d+) accepted")
def parse(f):
    out = []; pend = None
    for l in open(f, errors="replace"):
        t = T.match(l)
        if t: pend = (int(t.group(1)), int(t.group(2))); continue
        g = G.match(l)
        if g and pend: out.append((pend, float(g.group(1)))); pend = None
    return out
W = 100
def prof(d): return " ".join(f"{st.mean(d[i:i + W]):+.2f}" for i in range(0, len(d), W))
mode = sys.argv[1]
if mode == "pair":
    D = sys.argv[2]
    for r in sorted({int(re.search(r"-r(\d+)\.log", f).group(1)) for f in glob.glob(f"{D}/serve-*-r*.log")}):
        a, b = parse(f"{D}/serve-base-r{r}.log"), parse(f"{D}/serve-bk-r{r}.log"); n = min(len(a), len(b))
        al = sum(a[i][0] == b[i][0] for i in range(n)); d = [b[i][1] - a[i][1] for i in range(n)]
        print(f"r{r}: steps {len(a)}/{len(b)} aligned {al}; bk - base mean {st.mean(d):+.3f} ms; windows: {prof(d)}")
else:
    logs = sys.argv[2:]; S = {f: parse(f) for f in logs}; n = min(len(v) for v in S.values())
    ref = [st.median(S[f][i][1] for f in logs) for i in range(n)]
    for f in logs:
        al = sum(S[f][i][0] == S[logs[0]][i][0] for i in range(n)); d = [S[f][i][1] - ref[i] for i in range(n)]
        print(f"{f.split('/')[-1]:14s} steps {len(S[f])} (aligned {al}/{n}); mean offset from the group median {st.mean(d):+.3f} ms; windows: {prof(d)}")
