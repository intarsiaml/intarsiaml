#!/usr/bin/env python3
"""mcdiag: position-matched comparison of graph-step times between runs of the same greedy request (same ids, so the same accept pattern and
MoE routing): steps binned by the position they start at (`len` in the trace line, --bin tokens per bin). Prints the mean step ms per bin for the
base file(s) and the probe file(s), and their difference, plus the same summed over the bins before / after the first chain (--chain-pos POS; the
position of the first `spec: copy token` line of the first probe file when omitted).
usage: cmp.py --base F1 [F2..] --probe G1 [G2..] [--bin 50] [--chain-pos P]"""
import re, sys, argparse, collections

ap = argparse.ArgumentParser()
ap.add_argument("--base", nargs="+", required=True)
ap.add_argument("--probe", nargs="+", required=True)
ap.add_argument("--bin", type=int, default=50)
ap.add_argument("--chain-pos", type=int)
ap.add_argument("--skip-pos", type=int, default=100, help="ignore steps starting below this position (capture, warm-up)")
a = ap.parse_args()
G = re.compile(r"^spec: mtp graph step ([\d.]+) ms gap ([\d.]+) ms n (\d+) len (\d+) captures (\d+)")
C = re.compile(r"^spec: copy token \d+ pos (\d+)")


def load(fn):
    st, ch = [], None
    for l in open(fn, errors="replace"):
        m = G.match(l)
        if m:
            st.append((int(m.group(4)), float(m.group(1))))
        m = C.match(l)
        if m and ch is None:
            ch = int(m.group(1))
    return st, ch


def binned(fns):
    d = collections.defaultdict(list)
    for fn in fns:
        for pos, ms in load(fn)[0]:
            d[pos // a.bin].append(ms)
    return d


chain = a.chain_pos or load(a.probe[0])[1]
B, P = binned(a.base), binned(a.probe)
print(f"chain at pos {chain}; bins of {a.bin} positions; step ms (n steps)")
tot = collections.defaultdict(lambda: [0.0, 0])
for b in sorted(set(B) & set(P)):
    if b * a.bin < a.skip_pos:
        continue
    mb, mp = sum(B[b]) / len(B[b]), sum(P[b]) / len(P[b])
    side = "pre" if chain is not None and (b + 1) * a.bin <= chain else ("post" if chain is not None and b * a.bin > chain else "chain-bin")
    print(f"  pos {b * a.bin:4d}-{(b + 1) * a.bin - 1:4d} {side:9s} base {mb:7.3f} ({len(B[b]):3d})  probe {mp:7.3f} ({len(P[b]):3d})  diff {mp - mb:+.3f}")
    t = tot[side]
    t[0] += mp - mb
    t[1] += 1
for k, (s, n) in tot.items():
    print(f"mean diff over {k} bins: {s / n:+.3f} ms ({n} bins)")
