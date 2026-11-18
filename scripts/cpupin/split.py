#!/usr/bin/env python3
"""cpupin: does the post-chain offset live inside the request that ran the chain, or carry into later requests? For the n20-like arms of a real_an.py directory (serve-<arm>-r<round>.log[.gz]),
every 50-position bin of every request is compared with the same bin of the same request in the base arm of the round (as real_an.py), and classed:
A = no chain has run yet in the process, B = a chain ran earlier in THIS request (bin position past the request's first chain), C = no chain earlier in this request but one in an earlier request.
usage: split.py DIR ARM [BASE]   (defaults BASE=base)"""
import re, sys, glob, gzip, statistics as st, collections
d, arm = sys.argv[1], sys.argv[2]; base = sys.argv[3] if len(sys.argv) > 3 else "base"
G = re.compile(r"^spec: mtp graph step ([\d.]+) ms gap ([\d.]+) ms n (\d+) len (\d+)"); C = re.compile(r"^spec: copy token \d+ pos (\d+)")
opn = lambda f: gzip.open(f, "rt", errors="replace") if f.endswith(".gz") else open(f, errors="replace")
def parse(fn):
    reqs, cur, last = [], None, -1
    for l in opn(fn):
        g = G.match(l)
        if g:
            pos = int(g.group(4))
            if cur is None or pos < last - 50:
                cur = {"steps": [], "chains": []}; reqs.append(cur)
            last = pos; cur["steps"].append((pos, float(g.group(1)))); continue
        c = C.match(l)
        if c and cur is not None: cur["chains"].append(int(c.group(1)))
    return reqs[2:]  # drop the 2 warm-ups
def bins(steps):
    b = collections.defaultdict(list)
    for p, ms in steps: b[p // 50].append(ms)
    return {k: (sum(v) / len(v), len(v)) for k, v in b.items()}
out = {"A": [], "B": [], "C": []}; per = collections.defaultdict(lambda: {"A": [], "B": [], "C": []})
for fn in sorted(glob.glob(f"{d}/serve-{arm}-r*.log*")):
    r = int(re.search(r"-r(\d+)\.log", fn).group(1)); fb = glob.glob(f"{d}/serve-{base}-r{r}.log*")
    if not fb: continue
    ra, rb = parse(fn), parse(fb[0]); seen = 0
    for qa, qb in zip(ra, rb):
        A, B = bins(qa["steps"]), bins(qb["steps"]); first = min(qa["chains"]) if qa["chains"] else None
        for k in A:
            if k not in B or k * 50 < 100 or k * 50 >= 800 or A[k][1] < 10 or B[k][1] < 10: continue
            cls = "A" if (seen == 0 and (first is None or k * 50 + 49 < first)) else ("B" if first is not None and k * 50 > first else "C")
            if cls == "A" and seen > 0: cls = "C"
            out[cls].append(A[k][0] - B[k][0]); per[r][cls].append(A[k][0] - B[k][0])
        seen += len(qa["chains"])
f = lambda x: f"{st.mean(x):+.3f} (n {len(x)}, sd {st.pstdev(x):.3f})" if x else "n/a"
print(f"{arm} vs {base}: A (no chain yet) {f(out['A'])}; B (chain earlier in this request) {f(out['B'])}; C (chain only in earlier requests) {f(out['C'])}")
for r, v in sorted(per.items()): print(f"   r{r}: A {f(v['A'])}; B {f(v['B'])}; C {f(v['C'])}")
