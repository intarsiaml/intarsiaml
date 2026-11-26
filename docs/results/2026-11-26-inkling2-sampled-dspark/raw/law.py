#!/usr/bin/env python3
"""Lane ink-sampled: compare the two arms of tests/inkling2_sampled_real.rs (lines `T seed ids...`).

  law.py OFF.txt ON.txt [--shared-max S] [--law-on-min 1000000]

Seed equality: every seed present in both files (the shared seeds) must give the same ids (seed-exact rule).
Law: per temperature and per generated position 2.., Pearson chi-square of homogeneity between the off arm's seeds below
`--law-on-min` and the on arm's seeds at or above it (disjoint seed sets), cells with expected count < 5 pooled; p from the
regularized upper incomplete gamma (no scipy). Also Fisher's combination over the positions of one temperature.
"""
import math
import sys
from collections import Counter, defaultdict


def gammq(a, x):
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:
        ap, s = a, 1.0 / a
        d = s
        for _ in range(10000):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - gln))
    tiny = 1e-300
    b = x + 1 - a
    c = 1 / tiny
    d = 1 / b
    h = d
    for i in range(1, 10000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        de = d * c
        h *= de
        if abs(de - 1) < 1e-15:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi2(a, b):
    """2 x C homogeneity: tokens with pooled expected >= 5 in the smaller arm are cells; the rest go to 16 bins by id % 16 (a fixed
    function of the token); bins still below 5 join one remainder cell. Same rule as tests/ink_sampled/mod.rs."""
    na, nb = sum(a.values()), sum(b.values())
    n = na + nb
    small = lambda tot: tot * min(na, nb) / n < 5
    cells, bins = [], [[0, 0] for _ in range(16)]
    for k in set(a) | set(b):
        o = (a.get(k, 0), b.get(k, 0))
        if small(o[0] + o[1]):
            bins[k % 16][0] += o[0]
            bins[k % 16][1] += o[1]
        else:
            cells.append(o)
    rest = [0, 0]
    for bn in bins:
        if small(bn[0] + bn[1]):
            rest[0] += bn[0]
            rest[1] += bn[1]
        else:
            cells.append(tuple(bn))
    if rest[0] + rest[1]:
        cells.append(tuple(rest))
    stat = 0.0
    for o in cells:
        tot = o[0] + o[1]
        e = (tot * na / n, tot * nb / n)
        stat += (o[0] - e[0]) ** 2 / e[0] + (o[1] - e[1]) ** 2 / e[1]
    df = max(len(cells) - 1, 0)
    return stat, df, (gammq(df / 2, stat / 2) if df else 1.0)


def load(path):
    out = {}
    for line in open(path):
        f = line.split()
        if len(f) < 3:
            continue
        out[(f[0], int(f[1]))] = tuple(int(x) for x in f[2:])
    return out


def main():
    args = sys.argv[1:]
    on_min = 1000000
    if "--law-on-min" in args:
        i = args.index("--law-on-min")
        on_min = int(args[i + 1])
        del args[i : i + 2]
    off, on = load(args[0]), load(args[1])
    shared = sorted(set(off) & set(on))
    by_t = defaultdict(lambda: [0, 0])
    for k in shared:
        by_t[k[0]][0] += 1
        by_t[k[0]][1] += off[k] == on[k]
    for t, (n, eq) in sorted(by_t.items()):
        print(f"seed equality T {t}: {eq} / {n} shared seeds give identical ids")
    temps = sorted({k[0] for k in off} | {k[0] for k in on})
    for t in temps:
        a_runs = [v for k, v in off.items() if k[0] == t and k[1] < on_min]
        b_runs = [v for k, v in on.items() if k[0] == t and k[1] >= on_min]
        if not a_runs or not b_runs:
            continue
        L = min(min(map(len, a_runs)), min(map(len, b_runs)))
        fisher, npos = 0.0, 0
        for p in range(1, L):
            a = Counter(r[p] for r in a_runs)
            b = Counter(r[p] for r in b_runs)
            s, df, pv = chi2(a, b)
            fisher += -2 * math.log(max(pv, 1e-300))
            npos += 1
            print(f"law T {t} position {p + 1}: chi2 {s:.2f} df {df} p {pv:.4f} (n {len(a_runs)} off + {len(b_runs)} on; distinct {len(a)} / {len(b)})")
        if npos:
            print(f"law T {t} Fisher over {npos} positions: X2 {fisher:.2f} df {2 * npos} p {gammq(npos, fisher / 2):.4f}")


if __name__ == "__main__":
    main()
