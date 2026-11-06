"""G-tests of the statistical exactness gate (docs/results/2026-11-05-sampled-spec.md, rule 2).

usage: uv run --no-project --with scipy python gtest.py <runs dir>
Reads <cfg>-<arm>-<seed0>.txt (lines: seed t1 t2 t3 t4) and the .log telemetry lines; prints, per config, variant and
generated position 2..4, the G-test of homogeneity against spec-off (cells with expected count < 5 in either arm pooled),
its df, p-value, the Bonferroni pass, and the power against Cohen's w = 0.1 at that alpha.
"""
import collections, glob, os, re, sys
from scipy.stats import chi2, ncx2

d = sys.argv[1]
ALPHA = 0.01 / 12
W = 0.1
runs = collections.defaultdict(list)
tel = collections.defaultdict(lambda: [0, 0, 0])
for f in glob.glob(os.path.join(d, "*.txt")):
    cfg, arm, _ = os.path.basename(f)[:-4].split("-")
    for line in open(f):
        x = line.split()
        if len(x) == 5:
            runs[(cfg, arm)].append(tuple(int(t) for t in x[1:]))
    log = f[:-4] + ".log"
    if os.path.exists(log):
        m = re.search(r"spec steps (\d+) proposed (\d+) accepted (\d+)", open(log).read())
        if m:
            for i in range(3):
                tel[(cfg, arm)][i] += int(m.group(i + 1))


def gtest(a, b):
    ca, cb = collections.Counter(a), collections.Counter(b)
    na, nb = len(a), len(b)
    keys = set(ca) | set(cb)
    cells, pool = [], [0, 0]
    for k in keys:
        o = (ca[k], cb[k])
        tot = o[0] + o[1]
        if min(tot * na / (na + nb), tot * nb / (na + nb)) < 5:
            pool[0] += o[0]
            pool[1] += o[1]
        else:
            cells.append(o)
    if sum(pool) > 0:
        cells.append(tuple(pool))
    g = 0.0
    n = na + nb
    import math
    for o in cells:
        tot = o[0] + o[1]
        for j, nj in enumerate((na, nb)):
            e = tot * nj / n
            if o[j] > 0:
                g += 2 * o[j] * math.log(o[j] / e)
    df = len(cells) - 1
    p = chi2.sf(g, df)
    crit = chi2.ppf(1 - ALPHA, df)
    power = 1 - ncx2.cdf(crit, df, n * W * W)
    return g, df, p, power, len(cells)


print(f"alpha per test {ALPHA:.2e} (0.01 / 12), power against Cohen's w = {W}")
print("| config | variant | n off / n spec | acceptance | position | G | df | p | pass | power |")
print("|---|---|---|---|---:|---:|---:|---:|---|---:|")
allpass = True
for cfg in ("A", "B"):
    off = runs.get((cfg, "0"), [])
    for arm in ("reject", "match"):
        sp = runs.get((cfg, arm), [])
        if not off or not sp:
            continue
        steps, prop, acc = tel[(cfg, arm)]
        accs = f"{acc}/{prop} = {acc / prop:.3f}" if prop else "-"
        for pos in (2, 3, 4):
            g, df, p, power, _ = gtest([r[pos - 1] for r in off], [r[pos - 1] for r in sp])
            ok = p > ALPHA
            allpass &= ok
            print(f"| {cfg} | {arm} | {len(off)} / {len(sp)} | {accs} | {pos} | {g:.1f} | {df} | {p:.3f} | {'PASS' if ok else 'FAIL'} | {power:.3f} |")
        # joint (t2, t3, t4) as a supplementary line (not part of the pass rule)
        g, df, p, power, _ = gtest([r[1:] for r in off], [r[1:] for r in sp])
        print(f"| {cfg} | {arm} | | | joint 2-4 (info) | {g:.1f} | {df} | {p:.3f} | - | {power:.3f} |")
print("ALL PASS" if allpass else "SOME FAIL")
