#!/usr/bin/env python3
"""Lane ink-metal-dtopk: GPU clock and power during each D3 run (macmon pipe -i 250), grouped to separate the arm effect from endor's
power-state steps (docs memory: endor is a 140 W MacBook Pro; the GPU's burst clock depends on recent power history).
Run windows: end = the `<arm>@<rep> rc 0 HH:MM:SS` line of the slot log, start = end minus the run's own wall time (its .err
`real` seconds from /usr/bin/time -l). Prints per run: tok/s, median GPU MHz while busy (usage > 50 %), mean GPU W.
  power.py slotD.log macmon.jsonl D/d"""
import datetime as dt, json, re, statistics, sys
from pathlib import Path
log, mm, d = sys.argv[1], sys.argv[2], Path(sys.argv[3])
samples = []
for l in open(mm):
    try:
        r = json.loads(l)
    except ValueError:
        continue
    t = dt.datetime.fromisoformat(r["timestamp"]).astimezone()
    samples.append((t.replace(tzinfo=None), r["gpu_usage"][0], r["gpu_usage"][1], r["gpu_power"]))
day = samples[0][0].date()
dec = re.compile(r"\(([\d.]+) tok/s\)")
rows = []
for l in open(log):
    m = re.match(r"^(\w+)@(r\d) rc 0 (\d\d:\d\d:\d\d) .*decode overall:.*\(([\d.]+) tok/s\)", l)
    if not m:
        continue
    arm, rep, hms, tps = m.group(1), m.group(2), m.group(3), float(m.group(4))
    end = dt.datetime.combine(day, dt.time.fromisoformat(hms))
    real = next((float(x.split()[0]) for x in open(d / f"{arm}@{rep}.err", errors="replace") if x.strip().endswith(" real") or " real " in x), None)
    real = real or 50.0
    start = end - dt.timedelta(seconds=real)
    w = [s for s in samples if start <= s[0] <= end]
    busy = [s for s in w if s[2] > 0.5]
    mhz = statistics.median(s[1] for s in busy) if busy else float("nan")
    watt = statistics.mean(s[3] for s in w) if w else float("nan")
    rows.append((arm, rep, tps, mhz, watt, len(w)))
print("| run | tok/s | GPU MHz (median, busy) | GPU W (mean) | samples |\n|---|---:|---:|---:|---:|")
for a, r, t, m, w, n in rows:
    print(f"| {a}@{r} | {t:.2f} | {m:.0f} | {w:.1f} | {n} |")
