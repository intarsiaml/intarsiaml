import re, sys, glob, math, random, collections, json
O = sys.argv[1]
random.seed(7)
def parse(f):
    t = open(f).read()
    dec = {}
    for m in re.finditer(r"^decode\[(\d+)\|(\w+)\|c(\d+)\]: (\d+) tokens, (\d+) ms \(([0-9.]+) tok/s\)", t, re.M):
        dec[(int(m[1]), m[2], int(m[3]))] = float(m[6])
    ids = {}
    for m in re.finditer(r"^ids\[(\d+)\|(\w+)\|c(\d+)\]: (\[.*\])", t, re.M):
        ids[(int(m[1]), m[2], int(m[3]))] = m[4]
    sp = {}
    prev = (0, 0, 0)
    for m in re.finditer(r"^spec\[(\d+)\|(\w+)\|c(\d+)\]: steps (\d+) proposed (\d+) accepted (\d+)", t, re.M):
        cur = (int(m[4]), int(m[5]), int(m[6]))
        sp[(int(m[1]), m[2], int(m[3]))] = tuple(a - b for a, b in zip(cur, prev)); prev = cur
    return dec, ids, sp
def offids(cls):
    t = open(f"{O}/off.{cls}.err").read()
    return {int(m[1]): m[2] for m in re.finditer(r"^ids\[(\d+)\]: (\[.*\])", t, re.M)}
def boot(xs, f, n=4000):
    out = []
    for _ in range(n):
        out.append(f([random.choice(xs) for _ in xs]))
    out.sort(); return out[int(0.025 * n)], out[int(0.975 * n)]
def gm(x): return math.exp(sum(math.log(v) for v in x) / len(x))
def med(x): s = sorted(x); n = len(s); return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
print("| class | pairs | tok/s A (gate 0.8 k3) | tok/s B (cut 0.5 k3) | B/A median [95% CI] | B/A geomean [95% CI] | cluster-bootstrap geomean CI | pairs > 1.0 | prompts >= +2% (median per prompt) | prompts below 0.99 | min per-prompt median |")
print("|---|---:|---:|---:|---|---|---|---:|---|---|---|")
allids = []
accs = {}
for cls in ("short", "len1k", "len4k", "len16k"):
    pairs = []  # (pass, cycle, prompt, a, b)
    for f in sorted(glob.glob(f"{O}/ab.p*.{cls}.err")):
        p = int(re.search(r"\.p(\d+)\.", f)[1])
        dec, ids, sp = parse(f)
        off = offids(cls) if glob.glob(f"{O}/off.{cls}.err") else {}
        for (pi, arm, c), v in dec.items():
            if arm != "g3" or c == 0: continue
            b = dec.get((pi, "c5", c))
            if b: pairs.append((p, c, pi, v, b))
        for k, v in ids.items():
            allids.append((cls, k, v == off.get(k[0])))
        for k, v in sp.items():
            a = accs.setdefault((cls, k[1]), [0, 0, 0]); [a.__setitem__(i, a[i] + v[i]) for i in range(3)]
    if not pairs: print(f"| {cls} | 0 |"); continue
    r = [b / a for *_, a, b in pairs]
    A = med([a for *_, a, b in pairs]); B = med([b for *_, a, b in pairs])
    mlo, mhi = boot(r, med); glo, ghi = boot(r, gm)
    cl = collections.defaultdict(list)
    for p, c, pi, a, b in pairs: cl[(p, c)].append(b / a)
    keys = list(cl)
    def clg(ks): return gm([x for k in ks for x in cl[k]])
    clo, chi = boot(keys, clg, 2000)
    bp = collections.defaultdict(list)
    for p, c, pi, a, b in pairs: bp[pi].append(b / a)
    pm = {k: med(v) for k, v in bp.items()}
    n2 = sum(1 for v in pm.values() if v >= 1.02); nlow = sum(1 for v in pm.values() if v < 0.99)
    print(f"| {cls} | {len(r)} | {A:.1f} | {B:.1f} | {med(r):.3f} [{mlo:.3f}-{mhi:.3f}] | {gm(r):.3f} [{glo:.3f}-{ghi:.3f}] | [{clo:.3f}-{chi:.3f}] | {sum(1 for v in r if v > 1)}/{len(r)} | {n2}/{len(pm)} | {nlow}/{len(pm)} | {min(pm.values()):.3f} |")
print()
bad = [x for x in allids if not x[2]]
print(f"ids of {len(allids)} requests (both arms, all cycles incl. warm-up) vs --spec off: {len(allids) - len(bad)} identical, {len(bad)} differ")
for x in bad[:10]: print("  DIFF", x[:2])
print()
for (cls, arm), a in sorted(accs.items()):
    print(f"{cls} {arm}: verify passes {a[0]}, proposed {a[1]}, accepted {a[2]} ({100 * a[2] / max(a[1], 1):.0f}%), {(a[0] + a[2]) / max(a[0], 1):.2f} tok/pass")
