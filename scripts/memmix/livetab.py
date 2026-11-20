#!/usr/bin/env python3
"""memmix: glibc's own live-versus-free split over a run, from the malloc_info() blocks libmemstat.c prints into the serve log.
usage: livetab.py <raw-dir> <tag> [--skip MIN] [--md]     (needs memstat-<tag>.txt or serve-<tag>.log with the XML, memstat.py beside this file, mem-<tag>.csv, marks-<tag>.tsv)
Per block (every MEMSTAT_PERIOD s and around every trim): minutes of load, kind, arena address space, read-write part of the arenas (resident upper bound),
free chunks in the arenas (logical: malloc_trim(0) returns their pages to the OS but they stay free chunks), LIVE = rw part - free chunks, directly mmapped chunks (MB), live total = LIVE + mmapped, the store's host tier at that moment
(nearest 30 s sample), and the live total minus the store tier ("non-store live heap"). Then slopes (MB/h +- SE) of the non-store live heap and of the free chunks
over the blocks from --skip minutes (default 15). Stdlib only."""
import csv, math, pathlib, subprocess, sys

raw = pathlib.Path(sys.argv[1]); tag = sys.argv[2]
skip = float(sys.argv[sys.argv.index("--skip") + 1]) if "--skip" in sys.argv else 15.0
md = "--md" in sys.argv
marks = {}
for line in open(raw / f"marks-{tag}.tsv"):
    t, name = line.strip().split("\t"); marks[name] = int(t)
t0 = marks["load"]
rows = []
for r in csv.DictReader(open(raw / f"mem-{tag}.csv")):
    try:
        rows.append((float(r["epoch"]), float(r["prefix_store_tier_bytes_tier__host__"]) / 1e6, float(r["st_RssAnon"])))
    except ValueError:
        pass
here = pathlib.Path(__file__).parent
ms = raw / f"memstat-{tag}.txt"  # reduce.py keeps memstat.py's table here once the XML is stripped from the serve log
txt = (ms.read_text().splitlines() if ms.exists() else subprocess.run([sys.executable, str(here / "memstat.py"), str(raw / f"serve-{tag}.log"), str(t0)], capture_output=True, text=True, check=True).stdout.splitlines())
blocks = []
for line in txt[2:]:
    p = line.split("\t")
    m, kind, n, asp, rw, free, live, mn, mmb = float(p[0]), p[1], int(p[2]), *[float(x) for x in p[3:7]], int(p[7]), float(p[8])
    t = t0 + m * 60
    near = [r for r in rows if abs(r[0] - t) <= 30]
    host = sum(r[1] for r in near) / len(near) if near else float("nan")
    anon = sum(r[2] for r in near) / len(near) if near else float("nan")
    blocks.append({"min": m, "kind": kind, "arenas": n, "rw": rw, "free": free, "live": live, "mmap": mmb, "tot": live + mmb, "host": host, "anon": anon, "nonstore": live + mmb - host})
hdr = ["min of load", "kind", "arenas", "arena rw part", "free chunks", "arena live", "mmapped", "live total", "store host tier", "non-store live heap", "RssAnon (nearest sample)"]
print(("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr)) if md else "\t".join(hdr))
for b in blocks:
    c = [f"{b['min']:.1f}", b["kind"], str(b["arenas"]), f"{b['rw']:.0f}", f"{b['free']:.0f}", f"{b['live']:.0f}", f"{b['mmap']:.0f}", f"{b['tot']:.0f}", f"{b['host']:.0f}", f"{b['nonstore']:.0f}", f"{b['anon']:.0f}"]
    print(("| " + " | ".join(c) + " |") if md else "\t".join(c))


def fit(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan"), float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    res = [y - (my + b * (x - mx)) for x, y in zip(xs, ys)]
    return b, math.sqrt(sum(e * e for e in res) / (n - 2) / sxx)


print("\nslopes (MB per hour +- SE of the least-squares fit)")
# a "before trim" block and the "after trim" block of the same trim are the same numbers (a trim does not change the chunk accounting), so the fits use
# the periodic blocks and the after-trim blocks only; each fit is also given from later start minutes (the committed size is what an un-trimmed process can hold)
for start in (skip, 60.0, 90.0):
    for label, sel in [("periodic + after-trim blocks", lambda b: b["kind"] in ("periodic", "after trim")), ("after-trim blocks", lambda b: b["kind"] == "after trim"), ("periodic blocks", lambda b: b["kind"] == "periodic")]:
        s = [b for b in blocks if b["min"] >= start and sel(b)]
        if len(s) < 4:
            continue
        for name, key in [("live total (arena live + mmapped)", "tot"), ("arena committed size (rw part)", "rw"), ("free chunks in arenas (logical, not resident)", "free")]:
            b, se = fit([x["min"] / 60 for x in s], [x[key] for x in s])
            print(f"  from minute {start:.0f}, {label} (n={len(s)}): {name}: {b:+.0f} +- {se:.0f}")
