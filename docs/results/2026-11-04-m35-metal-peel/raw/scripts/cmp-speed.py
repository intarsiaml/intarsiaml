#!/usr/bin/env python3
"""localeval reps per size and arm. usage: cmp-speed.py DIR. Class of a rep by its prompt length p: E = exposed (p mod 4096 in 2050..2057), C = peeled but not exposed
(p mod 256 in 2..9, p > 256, tail pass above 256 rows), N = unchanged by the peel. Reps keep their own p, since the nonce moves the length by a few tokens."""
import glob, re, statistics as st, sys
d = sys.argv[1]
rows = []
for f in sorted(glob.glob(f"{d}/*-localeval.*")):
    m0 = re.search(r"/(main|peel)-(\d)-(\d+)-localeval", f)
    if not m0: continue
    for l in open(f):
        m = re.search(r"(\d+) tok rep (\d)/\d: prompt=(\d+) ttft=([\d.]+)s prefill=(\d+) tok/s decode=([\d.]+)", l)
        if m: rows.append((m0[1], int(m0[2]), int(m[1]), int(m[3]), float(m[4]), float(m[6])))
rows = list(dict.fromkeys(rows))
def cls(p):
    if 2050 <= p % 4096 <= 2057: return "E"
    t = p % 256
    if 2 <= t <= 9 and p > 256 and p % 4096 > 256: return "C"
    return "N"
print("size  arm  run  prompt lengths / class / ttft")
for size in sorted({r[2] for r in rows}):
    for arm in ("main", "peel"):
        for n in sorted({r[1] for r in rows if r[0] == arm and r[2] == size}):
            s = [r for r in rows if r[0] == arm and r[1] == n and r[2] == size]
            print(f"{size:6d} {arm} {n} {[r[3] for r in s]} {''.join(cls(r[3]) for r in s)} ttft {[r[4] for r in s]} decode {[r[5] for r in s]}")
    for c in ("E", "C", "N"):
        a = [r for r in rows if r[2] == size and r[0] == "main" and cls(r[3]) == c]; b = [r for r in rows if r[2] == size and r[0] == "peel" and cls(r[3]) == c]
        if not (a and b): continue
        pre = lambda x: st.median(r[3] / r[4] for r in x)
        ta, tb = st.median(r[4] for r in a), st.median(r[4] for r in b)
        print(f"   {size} class {c}: n {len(a)}/{len(b)}  ttft median {ta:.3f} -> {tb:.3f} s ({tb - ta:+.3f} s, {tb / ta - 1:+.1%})  prefill {pre(a):.0f} -> {pre(b):.0f} tok/s ({pre(b) / pre(a) - 1:+.1%})  decode {st.median(r[5] for r in a):.1f} -> {st.median(r[5] for r in b):.1f}")
