#!/usr/bin/env python3
"""Lane T2: per-rank decode-step breakdown from a rocprofv3 kernel_trace.csv of a tp=2 run (one process, two GPUs).
usage: prof2.py kernel_trace.csv [skip_steps=8]
A step is the interval between consecutive `argmax` kernel ends on rank 0's agent (the only rank with the head); every
kernel whose start falls inside a step counts for that step on its own agent. Per agent: kernel-busy time per step, the
P2P all-reduce kernel (`ar2_kernel`) time (it includes waiting for the peer), and the top kernels by time."""
import csv, sys, collections, re

path = sys.argv[1]
skip = int(sys.argv[2]) if len(sys.argv) > 2 else 8
rows = []
with open(path) as f:
    for r in csv.DictReader(f):
        rows.append((int(r["Start_Timestamp"]), int(r["End_Timestamp"]), r["Agent_Id"], r["Kernel_Name"]))
rows.sort()


def short(n):
    n = n.replace("(anonymous namespace)::", "")
    n = re.sub(r"^void ", "", n)
    d = 0
    for i, ch in enumerate(n):
        if ch == "<":
            d += 1
        elif ch == ">":
            d -= 1
        elif ch == "(" and d == 0:
            n = n[:i]
            break
    return n.replace("ik_bf16", "bf16")[:60]


head_agent = collections.Counter(a for _, _, a, k in rows if "argmax" in k).most_common(1)[0][0]
ends = [e for _, e, a, k in rows if a == head_agent and "argmax" in k]
# steady state: drop warm-up steps; the last boundary closes the last step
ends = ends[skip:]
if len(ends) < 3:
    sys.exit(f"only {len(ends)} decode steps after skipping {skip}")
n = len(ends) - 1
t0, t1 = ends[0], ends[-1]
agents = sorted({a for _, _, a, _ in rows})
print(f"rank 0 (head) = {head_agent}; decode steps analysed: {n}; step wall {((t1 - t0) / n) / 1e3:.1f} us")
for a in agents:
    ks = [(s, e, k) for s, e, ag, k in rows if ag == a and t0 <= s < t1]
    busy = sum(e - s for s, e, _ in ks)
    ar = [(e - s) for s, e, k in ks if "ar2_kernel" in k]
    by = collections.defaultdict(lambda: [0, 0])
    for s, e, k in ks:
        by[short(k)][0] += 1
        by[short(k)][1] += e - s
    print(f"\n{a}{' (rank 0)' if a == head_agent else ''}: {len(ks) / n:.0f} launches/step, kernel-busy {busy / n / 1e3:.1f} us/step "
          f"({100 * busy / (t1 - t0):.0f} % of wall), all-reduce {len(ar) / n:.0f}/step {sum(ar) / n / 1e3:.1f} us/step "
          f"(median {sorted(ar)[len(ar) // 2] / 1e3 if ar else 0:.1f} us), idle {((t1 - t0) - busy) / n / 1e3:.1f} us/step")
    print(f"  {'kernel':60} {'n/step':>7} {'us/step':>9}")
    for k, (c, t) in sorted(by.items(), key=lambda x: -x[1][1])[:14]:
        print(f"  {k:60} {c / n:7.1f} {t / n / 1e3:9.1f}")
