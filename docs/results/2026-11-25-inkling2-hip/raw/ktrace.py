#!/usr/bin/env python3
"""Per-kernel GPU time of the steady decode steps in a rocprofv3 kernel trace (csv) of `intarsia generate`.
The last N decode steps are taken as the last N occurrences of the argmax kernel; time = End - Start per dispatch.
usage: ktrace.py trace.csv [steps=8]"""
import collections, csv, re, sys
rows = list(csv.DictReader(open(sys.argv[1])))
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 8
rows.sort(key=lambda r: int(r["Start_Timestamp"]))
def short(n):
    n = n.replace("(anonymous namespace)::", "")
    n = re.sub(r"\(.*$", "", n)
    n = re.sub(r"^void ", "", n)
    return n.replace(" [clone .kd]", "")[:70]
marks = [i for i, r in enumerate(rows) if "argmax" in r["Kernel_Name"]]
if len(marks) <= steps:
    sys.exit(f"only {len(marks)} argmax dispatches")
lo, hi = marks[-steps - 1] + 1, marks[-1] + 1
seg = rows[lo:hi]
span = (int(seg[-1]["End_Timestamp"]) - int(rows[lo - 1]["End_Timestamp"])) / 1e3 / steps
g = collections.defaultdict(lambda: [0, 0.0])
for r in seg:
    k = short(r["Kernel_Name"])
    g[k][0] += 1
    g[k][1] += (int(r["End_Timestamp"]) - int(r["Start_Timestamp"])) / 1e3
busy = sum(v[1] for v in g.values()) / steps
print(f"steady decode: {steps} steps, {len(seg) / steps:.0f} dispatches / step, wall {span:.0f} us / step ({1e6 / span:.1f} tok/s), kernel time {busy:.0f} us / step ({100 * busy / span:.0f} %), gaps {span - busy:.0f} us")
print(f"{'kernel':70s} {'n/step':>7s} {'us/step':>9s} {'us/call':>8s} {'%wall':>6s}")
for k, (n, t) in sorted(g.items(), key=lambda kv: -kv[1][1])[:30]:
    print(f"{k:70s} {n / steps:7.1f} {t / steps:9.1f} {t / n:8.1f} {100 * t / steps / span:6.1f}")
