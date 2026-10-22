#!/usr/bin/env python3
"""Lane tc: aggregate the localeval speed arms of tc-window.sh (raw/<window>/loc-<arm>-r<N>.txt) into a table.

    python3 -I scripts/hip-decode/tc_speed.py <raw dir> [<raw dir> ...]

Each arm file has one row per prompt size ("| 16655 | 1.95 | 8549 (8280-8581) | ..."): TTFT s and the median prefill tok/s of
3 reps. localeval cuts its own prompt, so the token count differs by a few tokens from run to run; rows are grouped by size
class (about 4k: below 4200 tokens, about 16k: above 16000). Prints, per arm and class, the per-round values, their median
and mean."""
import glob
import re
import statistics as st
import sys

rows = {}
for d in sys.argv[1:]:
    for f in sorted(glob.glob(f"{d}/loc-*-r*.txt")):
        m = re.search(r"loc-(\w+)-r(\d+)\.txt$", f)
        if not m:
            continue
        arm, rnd = m.group(1), int(m.group(2))
        for line in open(f, errors="replace"):
            m2 = re.match(r"\| (\d+) \| ([\d.]+) \| (\d+) \(", line)
            if m2:
                n, ttft, pf = int(m2.group(1)), float(m2.group(2)), int(m2.group(3))
                cls = "4k" if n < 4200 else "16k"
                rows.setdefault((cls, arm), []).append((d.rstrip("/").split("/")[-1], rnd, n, ttft, pf))

for cls in ("4k", "16k"):
    for arm in ("base", "off", "on"):
        v = rows.get((cls, arm), [])
        if not v:
            continue
        pf = [x[4] for x in v]
        tt = [x[3] for x in v]
        print(
            f"{cls:>3} {arm:<4} n={len(v):<2} prefill tok/s median {st.median(pf):.0f} mean {st.mean(pf):.0f} "
            f"min {min(pf)} max {max(pf)} | TTFT s median {st.median(tt):.2f} | per round {pf}"
        )
