#!/usr/bin/env python3
"""ttft per localeval rep; each rep kept with its prompt length, since calibration moves the length between runs. Exposed = the peel changes the schedule."""
import glob, re, statistics as st
rows = []
for f in sorted(glob.glob(".scratch/logs/speed/*-[0-9]*-*-localeval.*")):
    m0 = re.search(r"/(main|peel)-(\d)-(\d+)-localeval", f)
    if not m0: continue
    for l in open(f):
        m = re.search(r"(\d+) tok rep (\d)/\d: prompt=(\d+) ttft=([\d.]+)s prefill=(\d+) tok/s decode=([\d.]+)", l)
        if m: rows.append((m0[1], int(m0[2]), int(m[1]), int(m[3]), float(m[4]), float(m[6])))
rows = list(dict.fromkeys(rows))
exposed = lambda p: 2 <= p % 256 <= 9 and p % 2048 >= 256
print("size  arm  rep prompts (residue mod 256) ttfts")
for size in sorted({r[2] for r in rows}):
    for arm in ("main", "peel"):
        for n in sorted({r[1] for r in rows if r[0] == arm and r[2] == size}):
            s = [r for r in rows if r[0] == arm and r[1] == n and r[2] == size]
            print(f"{size:6d} {arm} {n} prompts {[r[3] for r in s]} residues {[r[3] % 256 for r in s]} ttft {[r[4] for r in s]} decode {[r[5] for r in s]}")
    a = [r for r in rows if r[2] == size and r[0] == "main" and exposed(r[3])]; b = [r for r in rows if r[2] == size and r[0] == "peel" and exposed(r[3])]
    a2 = [r for r in rows if r[2] == size and r[0] == "main"]; b2 = [r for r in rows if r[2] == size and r[0] == "peel"]
    pre = lambda x: st.median(r[3] / r[4] for r in x)
    if a and b: print(f"   EXPOSED only: n {len(a)}/{len(b)}  ttft median {st.median(r[4] for r in a):.3f} -> {st.median(r[4] for r in b):.3f} s  ({st.median(r[4] for r in b) - st.median(r[4] for r in a):+.3f} s)  prefill {pre(a):.0f} -> {pre(b):.0f} tok/s ({pre(b) / pre(a) - 1:+.1%})  decode {st.median(r[5] for r in a):.1f} -> {st.median(r[5] for r in b):.1f}")
    print(f"   all reps: n {len(a2)}/{len(b2)} ttft median {st.median(r[4] for r in a2):.3f} -> {st.median(r[4] for r in b2):.3f}  prefill {pre(a2):.0f} -> {pre(b2):.0f} ({pre(b2) / pre(a2) - 1:+.1%})")
