#!/usr/bin/env python3
"""Queue wait vs run time of this round's fleetq jobs, from `fleetq log` on stdin:
  fleetq log | scripts/cuda-round5/queue-times.py [label substring ...]   (default: cuda-round5)
Lines used: '<MM-DD HH:MM:SS> #<id> queued by ...: <label>', '... #<id> started on ...: <label>', '... #<id> finished: <label>'."""
import re
import sys
from datetime import datetime

pats = sys.argv[1:] or ["cuda-round5"]
ev = {}
for line in sys.stdin:
    m = re.match(r"(\d\d-\d\d \d\d:\d\d:\d\d) #(\d+) (queued by \S+ on [^:]+|started on [^:]+|finished): (.*)", line.strip())
    if not m:
        continue
    t = datetime.strptime("2026-" + m.group(1), "%Y-%m-%d %H:%M:%S")
    kind = m.group(3).split()[0]
    label = m.group(4)
    if not any(p in label for p in pats):
        continue
    e = ev.setdefault(int(m.group(2)), {"label": label})
    e[kind] = t
wait = run = 0.0
print(f"{'job':>6} {'wait min':>8} {'run min':>8}  label")
for j in sorted(ev):
    e = ev[j]
    if "started" not in e or "finished" not in e:
        print(f"{j:>6} {'-':>8} {'-':>8}  {e['label'][:90]} (incomplete)")
        continue
    w = (e["started"] - e.get("queued", e["started"])).total_seconds() / 60
    r = (e["finished"] - e["started"]).total_seconds() / 60
    wait += w
    run += r
    print(f"{j:>6} {w:8.1f} {r:8.1f}  {e['label'][:90]}")
print(f"total: wait {wait:.1f} min, run {run:.1f} min over {len(ev)} jobs")
