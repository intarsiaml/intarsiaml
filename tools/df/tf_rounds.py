#!/usr/bin/env python3
"""Lane df: summarise TensorFold's per-request `done` lines (rounds, accepted/drafted, ms/round) by prompt size.
Usage: tf_rounds.py serve-tf.out [more files]. Only requests with tokens >= 128 count (warmups/calibration are 1 token)."""
import re, sys, statistics as st
from collections import defaultdict
pat = re.compile(r"done req-\S+ prompt=(\d+) .*?tokens=(\d+) .*?tok/s=([\d.]+) .*?rounds=(\d+) accepted=(\d+)/(\d+) ms/round=([\d.]+)")
for f in sys.argv[1:]:
    rows = defaultdict(list)
    for line in open(f, errors="replace"):
        m = pat.search(line)
        if not m:
            continue
        p, t, tps, r, a, d, ms = int(m[1]), int(m[2]), float(m[3]), int(m[4]), int(m[5]), int(m[6]), float(m[7])
        if t < 128 or r == 0:
            continue
        b = min((128, 1024, 4096, 16384), key=lambda s: abs(s - p))
        rows[b].append((t / r, ms, tps, d / r, a / max(d, 1)))
    print(f)
    print("| prompt | n | tokens/round | ms/round | tok/s | drafted/round | accept rate |")
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for b in sorted(rows):
        v = rows[b]
        med = lambda i: st.median(x[i] for x in v)
        print(f"| {b} | {len(v)} | {med(0):.2f} | {med(1):.1f} | {med(2):.1f} | {med(3):.1f} | {med(4):.3f} |")
