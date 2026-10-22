#!/usr/bin/env python3
"""Lane pf: tables from pf-stages.sh's window.log.

  pf_ab.py ab <window.log>   median / min / max prefill ms per arm and prompt size of the `ab` and `ab8` stages (reps interleaved),
                             with the change against the base binary's median (tok/s = prompt tokens / ms)
  pf_ab.py loc <loc-*.txt>   localeval prefill tok/s per size from the `loc` arms of pf-window.sh (median of the reps)
"""
import collections
import re
import statistics
import sys


def ab(path):
    rows = collections.defaultdict(list)  # (model, arm, size) -> [(prompt tokens, ms)]
    cur = None
    for line in open(path, errors="replace"):
        m = re.match(r"^(8b )?(base|dev\w+)-(\d+): ", line)
        if m:
            cur = (m.group(1) and "8B" or "35B", m.group(2), int(m.group(3)))
            continue
        fin = re.search(r"^finish=\S+ prompt=(\d+).*prefill (\d+) ms", line)
        if fin and cur:
            rows[cur].append((int(fin.group(1)), int(fin.group(2))))
            cur = None
    for model in ("35B", "8B"):
        print(f"\n{model}\n\n| prompt | arm | n | prefill ms median (min - max) | tok/s median | vs base |")
        print("|---|---|---|---|---|---|")
        sizes = sorted({k[2] for k in rows if k[0] == model})
        for size in sizes:
            base = None
            for arm in sorted({k[1] for k in rows if k[0] == model and k[2] == size}, key=lambda a: (a != "base", a)):
                v = rows[(model, arm, size)]
                ms = [x[1] for x in v]
                med = statistics.median(ms)
                tok = v[0][0]
                if arm == "base":
                    base = med
                delta = "" if base is None or arm == "base" else f"{100 * (base / med - 1):+.1f} %"
                print(f"| {size} ({tok} tokens) | {arm} | {len(ms)} | {med:.0f} ({min(ms)} - {max(ms)}) | {1000 * tok / med:.0f} | {delta} |")


def loc(paths):
    for p in paths:
        by = collections.defaultdict(list)
        for line in open(p, errors="replace"):
            m = re.search(r"\s(\d+) tok rep \d+/\d+: prompt=(\d+) ttft=([\d.]+)s prefill=(\d+) tok/s", line)
            if m:
                by[int(m.group(1))].append((int(m.group(4)), int(m.group(2))))
        print(p, {k: (statistics.median(x[0] for x in v), len(v)) for k, v in sorted(by.items())})


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    ab(sys.argv[2]) if sys.argv[1] == "ab" else loc(sys.argv[2:])
