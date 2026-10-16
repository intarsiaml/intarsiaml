#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Summarise a `macmon pipe` JSONL log (cbsoak_serve.sh writes macmon.jsonl): GPU clock (MHz), utilisation, power, temperature,
overall and per BIN-minute bin, plus the idle samples before the server started (first N samples) for the "was anything else
using the GPU" check.   uv run tools/macmon_summary.py macmon.jsonl [--bin-min 5] [--idle-n 5]"""
import argparse
import json
import statistics as st


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--bin-min", type=float, default=5.0)
    ap.add_argument("--idle-n", type=int, default=5)
    a = ap.parse_args()
    rows = []
    for ln in open(a.log):
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        rows.append((d["timestamp"], d["gpu_usage"][0], d["gpu_usage"][1] * 100, d["gpu_power"], d["temp"]["gpu_temp_avg"]))
    if not rows:
        print("no samples")
        return 1
    from datetime import datetime
    t0 = datetime.fromisoformat(rows[0][0])
    def line(name, rs):
        f = [r[1] for r in rs]; u = [r[2] for r in rs]; w = [r[3] for r in rs]; t = [r[4] for r in rs]
        return f"| {name} | {len(rs)} | {st.mean(f):.0f} | {min(f):.0f} | {q(f, 5):.0f} | {st.mean(u):.0f} | {q(u, 50):.0f} | {st.mean(w):.1f} | {max(w):.1f} | {max(t):.0f} |"
    print("| span | n | GPU MHz mean | min | p5 | util % mean | util % p50 | GPU W mean | max | max temp C |\n|---|---|---|---|---|---|---|---|---|---|")
    print(line(f"idle before server (first {a.idle_n})", rows[:a.idle_n]))
    print(line("whole log", rows))
    bins = {}
    for r in rows:
        bins.setdefault(int((datetime.fromisoformat(r[0]) - t0).total_seconds() // (a.bin_min * 60)), []).append(r)
    for k in sorted(bins):
        print(line(f"min {k * a.bin_min:.0f}-{(k + 1) * a.bin_min:.0f}", bins[k]))
    print(line("last 5 samples", rows[-5:]))


if __name__ == "__main__":
    main()
