#!/usr/bin/env python3
"""samp2s: per-window clock / memory / log summary from the raw dir: mean and min SM clock over the loaded samples (power > 25 W; this GB10's utilisation
field reads 0 for most samples, so the window.sh guard's util > 90 filter sees few), SW power cap delta, minimum MemAvailable, KILLED lines, warn/error lines.
  python3 scripts/samp2s/windows.py RAW_DIR a1 a2 a3 a4"""
import pathlib, statistics as st, sys
raw = pathlib.Path(sys.argv[1])
print("| window | loaded samples | mean SM MHz | min SM MHz | max W | SW cap delta (us) | min MemAvailable GB | killed | warn/error lines |\n|---|---|---|---|---|---|---|---|---|")
for w in sys.argv[2:]:
    rows = [l.split() for l in (raw / f"clk-{w}.tsv").read_text().splitlines() if len(l.split()) == 5]
    ld = [(int(r[1]), float(r[2])) for r in rows if float(r[2]) > 25]
    caps = [int(l.split("\t")[2]) for l in (raw / f"marks-{w}.tsv").read_text().splitlines() if len(l.split("\t")) == 3 and l.split("\t")[2].strip().lstrip("-").isdigit()]
    mem = [int(l.split()[1]) for l in (raw / f"mem-{w}.txt").read_text().splitlines() if len(l.split()) == 2 and l.split()[1].isdigit()]
    killed = sum("KILLED" in l for l in (raw / f"mem-{w}.txt").read_text().splitlines())
    log = (raw / f"serve-{w}.log").read_text().lower()
    warn = sum(any(k in l for k in ("warn", "error", "panic", "fail")) for l in log.splitlines())
    print(f"| {w} | {len(ld)} | {st.mean(c for c, _ in ld):.0f} | {min(c for c, _ in ld)} | {max(p for _, p in ld):.0f} | {(caps[-1] - caps[0]) if caps else '-'} | {min(mem) / 1024:.1f} | {killed} | {warn} |")
