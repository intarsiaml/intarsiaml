#!/usr/bin/env python3
"""memgrow analysis: python3 analyze.py <raw-dir> <tag> [--md]
Reads mem-<tag>.csv (scripts/memgrow/sampler.sh, 30 s rows) and marks-<tag>.tsv (phase marks: base start settle load idle stop after).
Prints: phase means, a table every 10 min of the load phase, store-full time, and least-squares slopes with standard errors (MB/h) of
MemAvailable, the driver-held residual, the engine's GPU memory (nvidia-smi per pid), RSS / PSS anon, the host tier bytes ...
over windows of the load phase, plus the same after dropping a settling period. Stdlib only."""
import csv, math, sys, pathlib

MEM_TOTAL_MB = 127600812 // 1024  # /proc/meminfo MemTotal of the GB10


def load(raw, tag):
    rows = list(csv.DictReader(open(raw / f"mem-{tag}.csv")))
    marks = {}
    for line in open(raw / f"marks-{tag}.tsv"):
        t, name = line.strip().split("\t")
        marks[name] = int(t)
    out = []
    for r in rows:
        d = {}
        for k, v in r.items():
            try:
                d[k] = float(v)
            except (TypeError, ValueError):
                d[k] = float("nan")
        # derived: driver-held residual = memory in use that no /proc category claims (cudaMalloc / pinned / driver on a unified-memory GB10)
        d["driver_held"] = MEM_TOTAL_MB - d["mi_MemFree"] - d["mi_Buffers"] - d["mi_Cached"] - d["mi_AnonPages"] - d["mi_Slab"] - d["mi_PageTables"] - d["mi_KernelStack"] - d["mi_Percpu"]
        d["host_tier_mb"] = d["prefix_store_tier_bytes_tier__host__"] / 1e6
        d["min"] = 0.0
        out.append(d)
    return out, marks


def fit(xs, ys):
    n = len(xs)
    if n < 4:
        return float("nan"), float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    res = [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]
    # residuals are autocorrelated (a 30 s series of a sawtooth), so also report an effective-n corrected SE: lag-1 autocorrelation inflation
    r1 = sum(res[i] * res[i - 1] for i in range(1, n)) / max(1e-9, sum(e * e for e in res))
    infl = math.sqrt(max(1.0, (1 + r1) / max(0.05, 1 - r1))) if r1 > 0 else 1.0
    se = math.sqrt(sum(e * e for e in res) / (n - 2) / sxx) * infl
    return b, se


def slope_mb_per_h(d, key, t0, t1):
    xs = [(r["epoch"] - t0) / 3600 for r in d if t0 <= r["epoch"] <= t1 and not math.isnan(r[key])]
    ys = [r[key] for r in d if t0 <= r["epoch"] <= t1 and not math.isnan(r[key])]
    return fit(xs, ys)


def ols2(rows, ykey, xkey, t0, t1):
    """y = a + b * x + c * t_hours by least squares over the rows in [t0, t1]; returns (c, se_c) in y-units per hour (se corrected like fit())."""
    pts = [(r[ykey], r[xkey], (r["epoch"] - t0) / 3600) for r in rows if t0 <= r["epoch"] <= t1 and not math.isnan(r[ykey]) and not math.isnan(r[xkey])]
    n = len(pts)
    if n < 8:
        return float("nan"), float("nan")
    X = [[1.0, x, t] for _, x, t in pts]; Y = [y for y, _, _ in pts]
    A = [[sum(X[k][i] * X[k][j] for k in range(n)) for j in range(3)] for i in range(3)]
    bvec = [sum(X[k][i] * Y[k] for k in range(n)) for i in range(3)]
    # solve 3x3 by Gauss-Jordan on [A | I]
    M = [A[i] + [1.0 if i == j else 0.0 for j in range(3)] for i in range(3)]
    for i in range(3):
        piv = max(range(i, 3), key=lambda r: abs(M[r][i])); M[i], M[piv] = M[piv], M[i]
        d = M[i][i]; M[i] = [v / d for v in M[i]]
        for r in range(3):
            if r != i:
                f = M[r][i]; M[r] = [a - f * b for a, b in zip(M[r], M[i])]
    inv = [row[3:] for row in M]
    beta = [sum(inv[i][j] * bvec[j] for j in range(3)) for i in range(3)]
    res = [Y[k] - sum(beta[i] * X[k][i] for i in range(3)) for k in range(n)]
    r1 = sum(res[i] * res[i - 1] for i in range(1, n)) / max(1e-9, sum(e * e for e in res))
    infl = math.sqrt(max(1.0, (1 + r1) / max(0.05, 1 - r1))) if r1 > 0 else 1.0
    s2 = sum(e * e for e in res) / (n - 3)
    return beta[2], math.sqrt(s2 * inv[2][2]) * infl


def mean(d, key, t0, t1):
    v = [r[key] for r in d if t0 <= r["epoch"] <= t1 and not math.isnan(r[key])]
    return sum(v) / len(v) if v else float("nan")


COLS = [  # (header, key, fmt)
    ("avail", "mi_MemAvailable", "{:.0f}"), ("held", "driver_held", "{:.0f}"), ("Cached", "mi_Cached", "{:.0f}"), ("Anon", "mi_AnonPages", "{:.0f}"),
    ("Shmem", "mi_Shmem", "{:.0f}"), ("Slab", "mi_Slab", "{:.0f}"),
    ("gpu_i", "gpu_mb_intarsia", "{:.0f}"), ("gpu_v", "gpu_mb_vllm", "{:.0f}"), ("devfree", "device_memory_free_bytes", None),
    ("RssAnon", "st_RssAnon", "{:.0f}"), ("RssFile", "st_RssFile", "{:.0f}"), ("PssAnon", "sm_Pss_Anon", "{:.0f}"), ("Pss", "sm_Pss", "{:.0f}"),
    ("host", "host_tier_mb", "{:.0f}"), ("hostobj", "prefix_store_tier_objects_tier__host__", "{:.0f}"), ("evict", "prefix_store_tier_evictions_total_tier__host__", "{:.0f}"),
    ("v_cg", "vllm_cg_current", "{:.0f}"), ("v_rss", "vllm_rss_engine", "{:.0f}"), ("reqs", "requests_total", "{:.0f}"),
]


def main():
    raw = pathlib.Path(sys.argv[1]); tag = sys.argv[2]
    d, m = load(raw, tag)
    for r in d:
        r["device_memory_free_bytes"] = r["device_memory_free_bytes"] / 1e6
    last = d[-1]["epoch"]
    for k in ("idle", "stop", "after"):
        m.setdefault(k, last)
    t_load, t_idle, t_stop = m["load"], m["idle"], m["stop"]
    full = next((r["epoch"] for r in d if r["epoch"] > m["start"] and r["host_tier_mb"] >= 0.95 * 8589.9), None)
    print(f"samples {len(d)}; phases (min from start): " + ", ".join(f"{k}={(v - m['start']) / 60:.1f}" for k, v in sorted(m.items(), key=lambda kv: kv[1])))
    print(f"host tier first >= 95% of its 8589.9 MB budget at +{(full - t_load) / 60:.1f} min of load" if full else "host tier never reached 95% of budget")
    print("\nphase means (MB)")
    print("phase\t" + "\t".join(c[0] for c in COLS))
    for name, a, b in [("base", m["base"], m["start"]), ("settle", m["settle"], m["load"]), ("idle", t_idle, t_stop), ("after", m["after"] - 120, m["after"])]:
        print(name + "\t" + "\t".join((f.format(mean(d, k, a, b)) if f else f"{mean(d, k, a, b):.0f}") for _, k, f in COLS))
    print("\nload phase, mean per 10 min (MB)")
    print("min\t" + "\t".join(c[0] for c in COLS))
    t = t_load
    while t < t_idle:
        e = min(t + 600, t_idle)
        print(f"{(t - t_load) / 60:.0f}-{(e - t_load) / 60:.0f}\t" + "\t".join((f.format(mean(d, k, t, e)) if f else f"{mean(d, k, t, e):.0f}") for _, k, f in COLS))
        t = e
    print("\nslopes, MB per hour +- SE (autocorrelation-corrected), over windows of the load phase")
    keys = ["mi_MemAvailable", "driver_held", "mi_Cached", "mi_AnonPages", "gpu_mb_intarsia", "gpu_mb_vllm", "device_memory_free_bytes", "st_RssAnon", "sm_Pss_Anon", "sm_Pss", "host_tier_mb", "vllm_cg_current", "vllm_rss_engine"]
    wins = [("all load", t_load, t_idle)]
    for skip in (10, 20, 30):
        wins.append((f"load, from +{skip} min", t_load + skip * 60, t_idle))
    mid = t_load + (t_idle - t_load) / 2
    wins += [("first half", t_load, mid), ("second half", mid, t_idle), ("idle phase", t_idle, t_stop)]
    print("window\t" + "\t".join(k for k in keys))
    for name, a, b in wins:
        cells = []
        for k in keys:
            s, se = slope_mb_per_h(d, k, a, b)
            cells.append(f"{s:+.0f}+-{se:.0f}")
        print(name + "\t" + "\t".join(cells))
    print("\nRssAnon slope with the host-tier sawtooth regressed out (RssAnon = a + b * host_tier + c * t), c in MB/h")
    for name, a, b in wins[:4] + wins[5:6]:
        c, se = ols2(d, "st_RssAnon", "host_tier_mb", a, b)
        c2, se2 = ols2(d, "mi_MemAvailable", "host_tier_mb", a, b)
        print(f"{name}\tRssAnon c = {c:+.0f} +- {se:.0f}\tMemAvailable c = {c2:+.0f} +- {se2:.0f}")
    print("\nbracket: MemAvailable / driver_held / gpu_vllm before the server (base) vs after it stopped (after)")
    for k in ("mi_MemAvailable", "driver_held", "gpu_mb_vllm", "mi_Cached", "mi_AnonPages", "vllm_cg_current"):
        print(f"{k}: base {mean(d, k, m['base'], m['start']):.0f}  after {mean(d, k, m['after'] - 120, m['after']):.0f}  delta {mean(d, k, m['after'] - 120, m['after']) - mean(d, k, m['base'], m['start']):+.0f}")


main()
