#!/usr/bin/env python3
"""Within-process A/B summary (lane ink-decroof): DIR/p*.err from the diag binary, meta_ab.json (cycle, length, arm) per prompt.
A all off, B CONV_PRE, C CONV_PRE + BIAS_PRE, D C + BIAS_EARLY. Cycle 0 is warm-up (dropped). A pair = two arms of the same
process, cycle and length."""
import glob, json, os, random, re, statistics as st, sys, math
O = sys.argv[1]
meta = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meta_ab.json')))
S = {}
for f in sorted(glob.glob(O + '/p*.err')):
    proc = re.match(r'p(\d+)\.err', os.path.basename(f)).group(1)
    dec = {}
    for line in open(f):
        m = re.match(r'decode\[(\d+)\]: \d+ tokens, \d+ ms \(([\d.]+) tok/s\)', line)
        if m: dec[int(m.group(1))] = float(m.group(2))
    for i, (c, L, a) in enumerate(meta):
        if i in dec: S[(proc, c, L, a)] = dec[i]
procs = sorted({k[0] for k in S}); arms = sorted({k[3] for k in S})
print(f'within-process, processes {procs}, arms {arms}, cycle 0 dropped (decode tok/s, 256 tokens)')
def cmp(L, x, y):
    rs = [S[(p, c, l, x)] / S[(p, c, l, y)] for (p, c, l, a) in S if l == L and a == x and c > 0 and (p, c, l, y) in S]
    if not rs: return
    random.seed(1)
    boots = sorted(st.median(random.choices(rs, k=len(rs))) for _ in range(4000))
    pos = sum(1 for r in rs if r > 1); n = len(rs)
    tail = sum(math.comb(n, k) for k in range(max(pos, n - pos), n + 1)) / 2 ** n
    per = [round(st.median(q), 4) for p in procs if (q := [S[(p, c, L, x)] / S[(p, c, L, y)] for c in range(1, 99) if (p, c, L, x) in S and (p, c, L, y) in S])]
    print(f'  {L} {x}/{y}: median {st.median(rs):.4f}, 95% CI [{boots[100]:.4f}-{boots[3899]:.4f}], min-max [{min(rs):.4f}-{max(rs):.4f}], {pos}/{n} above 1, sign p={min(1.0, 2 * tail):.3f}, per-process {per}')
for L in ('143', '1k', '4k'):
    for a in arms:
        xs = [v for k, v in S.items() if k[2] == L and k[3] == a and k[1] > 0]
        if xs: print(f'  {L} {a}: median {st.median(xs):.2f} [{min(xs):.2f}-{max(xs):.2f}] n={len(xs)}')
    for x, y in (('B', 'A'), ('C', 'A'), ('D', 'A'), ('C', 'B'), ('D', 'C')):
        cmp(L, x, y)
