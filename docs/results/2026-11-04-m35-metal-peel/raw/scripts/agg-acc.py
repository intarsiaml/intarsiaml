#!/usr/bin/env python3
"""usage: agg-acc.py BEFORE1.tsv AFTER1.tsv [BEFORE2.tsv AFTER2.tsv ...] -- pooled over the prompts the peel reached: paired mean-KL differences, sign counts, exact sign-test p, t, ratio of means."""
import sys, math, collections
from math import comb
def load(p):
    d = collections.OrderedDict()
    for l in open(p):
        f = l.rstrip("\n").split("\t")
        if len(f) >= 8: d.setdefault((f[0][:7], f[1]), []).append((float(f[5]), tuple(f[2:8])))
    return d
res = collections.defaultdict(list); rowsd = collections.defaultdict(list)
for bp, ap in zip(sys.argv[1::2], sys.argv[2::2]):
    B, A = load(bp), load(ap)
    for k in B:
        if B[k] != A[k]:
            kb = sum(x[0] for x in B[k]) / len(B[k]); ka = sum(x[0] for x in A[k]) / len(A[k])
            res[k[0]].append((k[1], kb, ka))
            for x, y in zip(B[k], A[k]):
                if x[1] != y[1]: rowsd[k[0]].append((x[0], y[0]))
for mode, R in res.items():
    n = len(R); dif = [kb - ka for _, kb, ka in R]
    m = sum(dif) / n; sd = (sum((v - m) ** 2 for v in dif) / (n - 1)) ** .5 if n > 1 else float("nan"); se = sd / n ** .5
    pos = sum(v > 0 for v in dif); neg = sum(v < 0 for v in dif); k = min(pos, neg)
    p = min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)
    mb = sum(kb for _, kb, _ in R) / n; ma = sum(ka for _, _, ka in R) / n
    rat = sorted(ka / kb for _, kb, ka in R); med = rat[n // 2] if n % 2 else (rat[n // 2 - 1] + rat[n // 2]) / 2
    print(f"{mode}: {n} prompts reached; mean KL before {mb:.4e} after {ma:.4e} (after/before {ma / mb:.3f}, median per-prompt ratio {med:.3f}); paired diff (before - after) {m:+.3e} +- {se:.1e} se (t {m / se:+.2f}); after closer on {pos}, farther on {neg}; sign-test p {p:.3f}")
    rr = rowsd[mode]; cl = sum(a > b for a, b in rr); fa = sum(a < b for a, b in rr); lr = sorted(math.log(b / a) for a, b in rr if a > 0 and b > 0); ml = lr[len(lr) // 2] if lr else float("nan")
    print(f"   rows that differ: {len(rr)}; after closer on {cl}, farther on {fa}; median per-row ratio after/before {math.exp(ml):.3f} (rows of one prompt are correlated, so this is descriptive, not a test)")
    print("   per prompt (before, after):", ", ".join(f"{nm}:{kb:.2e}/{ka:.2e}" for nm, kb, ka in R))
