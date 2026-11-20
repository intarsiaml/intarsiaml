#!/usr/bin/env python3
"""memmix: table of the malloc_trim(0) events of a run and the slopes of the before / after floors.
usage: trimtab.py <raw-dir> <tag> [--md] [--skip MIN]
Reads memtrim-<tag>.txt (the '[memtrim]' lines of the serve log, libmemstat.c: RssAnon / VmRSS before and after each trim, read in the engine itself),
mem-<tag>.csv (30 s sampler rows: store host tier bytes, MemAvailable, engine GPU) and marks-<tag>.tsv (the 'load' mark).
Per trim: minutes of load, RssAnon before / after, MB handed back, the store's host tier at that moment (nearest sample), the non-store anon before / after
(RssAnon minus host tier), and the 10-minute mean of MemAvailable around it. Then least-squares slopes (MB/h, +- SE) of: RssAnon before, RssAnon after,
non-store anon before, non-store anon after, over the trims from --skip minutes of load on (default 15: the store fill is over). Stdlib only."""
import csv, math, re, sys, pathlib

raw = pathlib.Path(sys.argv[1]); tag = sys.argv[2]
md = "--md" in sys.argv
skip = float(sys.argv[sys.argv.index("--skip") + 1]) if "--skip" in sys.argv else 15.0
marks = {}
for line in open(raw / f"marks-{tag}.tsv"):
    t, name = line.strip().split("\t"); marks[name] = int(t)
rows = []
for r in csv.DictReader(open(raw / f"mem-{tag}.csv")):
    try:
        rows.append((float(r["epoch"]), float(r["mi_MemAvailable"]), float(r["st_RssAnon"]), float(r["prefix_store_tier_bytes_tier__host__"]) / 1e6, float(r["gpu_mb_intarsia"])))
    except ValueError:
        pass
ev = []
for line in open(raw / f"memtrim-{tag}.txt"):
    d = dict(re.findall(r"(\w+)=(\S+)", line))
    ev.append({k: float(v) for k, v in d.items()})
# the supervisor (also preloaded) never trims here: only the engine's trigger file is touched, so one pid
t_load = marks["load"]


def near(t, col, w=30):
    c = [r for r in rows if abs(r[0] - t) <= w]
    return sum(r[col] for r in c) / len(c) if c else float("nan")


def win(t0, t1, col):
    c = [r[col] for r in rows if t0 <= r[0] < t1]
    return sum(c) / len(c) if c else float("nan")


def fit(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    res = [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]
    se = math.sqrt(sum(e * e for e in res) / (n - 2) / sxx) if n > 2 else float("nan")
    return b, se


out = []
for e in ev:
    t = e["t"]
    host = near(t, 3)
    b, a = e["before_rssanon_kb"] / 1024, e["after_rssanon_kb"] / 1024
    out.append({"min": (t - t_load) / 60, "before": b, "after": a, "freed": b - a, "host": host, "nsb": b - host, "nsa": a - host,
                "avail": win(t - 300, t + 300, 1), "trim_ms": e["trim_ms"], "t": t})
hdr = ["min of load", "RssAnon before", "RssAnon after", "handed back", "store host tier", "non-store before", "non-store after", "MemAvailable (10 min mean)", "trim ms"]
if md:
    print("| " + " | ".join(hdr) + " |"); print("|" + "---|" * len(hdr))
else:
    print("\t".join(hdr))
for o in out:
    cells = [f"{o['min']:.1f}", f"{o['before']:.0f}", f"{o['after']:.0f}", f"{o['freed']:.0f}", f"{o['host']:.0f}", f"{o['nsb']:.0f}", f"{o['nsa']:.0f}", f"{o['avail']:.0f}", f"{o['trim_ms']:.0f}"]
    print(("| " + " | ".join(cells) + " |") if md else "\t".join(cells))
sel = [o for o in out if o["min"] >= skip]
print(f"\nslopes over {len(sel)} trims from minute {skip:.0f} of load (MB per hour, +- SE of the least-squares fit; the last 'trim' at the end of the load is included)")
for name, key in [("RssAnon before trim", "before"), ("RssAnon after trim", "after"), ("non-store anon before", "nsb"), ("non-store anon after (the floor)", "nsa"), ("MemAvailable (10 min mean)", "avail")]:
    b, se = fit([o["min"] / 60 for o in sel], [o[key] for o in sel])
    print(f"  {name}: {b:+.0f} +- {se:.0f}")

# regress the store's host tier out (RssAnon = a + b * host tier + c * t): c is the drift that is not the store's sawtooth
def ols2(pts):
    n = len(pts)
    if n < 6:
        return float("nan"), float("nan")
    X = [[1.0, h, t] for _, h, t in pts]; Y = [y for y, _, _ in pts]
    A = [[sum(X[k][i] * X[k][j] for k in range(n)) for j in range(3)] for i in range(3)]
    bv = [sum(X[k][i] * Y[k] for k in range(n)) for i in range(3)]
    M = [A[i] + [1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for i in range(3):
        piv = max(range(i, 3), key=lambda r: abs(M[r][i])); M[i], M[piv] = M[piv], M[i]
        d = M[i][i]; M[i] = [v / d for v in M[i]]
        for r in range(3):
            if r != i:
                f = M[r][i]; M[r] = [a - f * b for a, b in zip(M[r], M[i])]
    inv = [row[3:] for row in M]
    beta = [sum(inv[i][j] * bv[j] for j in range(3)) for i in range(3)]
    res = [Y[k] - sum(beta[i] * X[k][i] for i in range(3)) for k in range(n)]
    s2 = sum(e * e for e in res) / (n - 3)
    return beta[2], math.sqrt(s2 * inv[2][2])


print("with the store's host tier regressed out (RssAnon = a + b * host tier + c * t), c in MB/h:")
for name, key in [("before trim", "before"), ("after trim", "after")]:
    c, se = ols2([(o[key], o["host"], o["min"] / 60) for o in sel])
    print(f"  RssAnon {name}: c = {c:+.0f} +- {se:.0f}")
if out:
    print(f"first / last floor (non-store anon after trim): {out[0]['nsa']:.0f} / {out[-1]['nsa']:.0f} MB; mean handed back {sum(o['freed'] for o in out) / len(out):.0f} MB")
