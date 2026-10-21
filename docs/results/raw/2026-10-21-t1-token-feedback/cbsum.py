#!/usr/bin/env python3
"""cbsum.py <gates-dir>: per context and arm, the medians over runs of cbgap.py's step period, GPU busy and idle gap
(cb traces cb-<ctx>-<arm>-<round>-<n>.jsonl.gz, last 128 decode steps of each)."""
import glob, os, re, statistics as st, subprocess, sys

D = sys.argv[1]
here = os.path.dirname(os.path.abspath(__file__))
rows = {}
for f in sorted(glob.glob(os.path.join(D, "cb-*.jsonl.gz"))):
    ctx, arm = re.match(r"cb-([^-]+)-([^-]+)-", os.path.basename(f)).groups()
    out = subprocess.run([sys.executable, os.path.join(here, "cbgap.py"), f, "128"], capture_output=True, text=True).stdout
    m = re.search(r"period ([\d.]+) ms .*busy ([\d.]+)\s+idle gap ([-\d.]+)", out)
    b = re.search(r"buffers/step (\d+)", out)
    rows.setdefault((ctx, arm), []).append((float(m.group(1)), float(m.group(2)), float(m.group(3)), int(b.group(1))))
print(f"{'ctx':6} {'arm':5} {'runs':>4} {'period ms':>10} {'busy ms':>8} {'gap ms':>7} {'tok/s':>6} buffers/step  periods")
for (ctx, arm), r in sorted(rows.items()):
    md = lambda i: st.median(x[i] for x in r)
    print(f"{ctx:6} {arm:5} {len(r):4d} {md(0):10.3f} {md(1):8.3f} {md(2):7.3f} {1e3 / md(0):6.2f} {md(3):5.0f}         "
          + " ".join(f"{x[0]:.2f}" for x in r))
