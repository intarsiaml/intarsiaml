#!/usr/bin/env python3
"""cpupin: thread residency of the engine's busiest thread (the launch thread) from the tsample.py logs (ts-<arm>-r<round>.tsv) of scripts/cpupin/ab.sh, split at the
first copy-chain pass of the server (`spec: copy pass ... at <unix time>` in serve-<arm>-r<round>.log; arms without chains: no split). BIG = cpu 5-9,15-19 (Cortex-X925),
everything else LITTLE (A725). Prints per server: the thread, cpu ticks, share of its ticks on BIG before / after the first chain, migrations between BIG and LITTLE per
10 s of busy time, and the cpus used. usage: tsan.py DIR"""
import re, sys, glob, collections
BIG = set(range(5, 10)) | set(range(15, 20))
d = sys.argv[1]
print(f"{'server':12s} {'thread':16s} {'ticks':>6s} {'chain@':>7s} {'BIG% before':>12s} {'BIG% after':>11s} {'big<->little moves/10s busy':>28s}  cpus(after)")
for fn in sorted(glob.glob(f"{d}/ts-*-r*.tsv")):
    m = re.search(r"ts-(.+-r\d+)\.tsv$", fn); name = m.group(1)
    t0 = None
    try:
        for l in open(f"{d}/serve-{name}.log", errors="replace"):
            c = re.match(r"^spec: copy pass .* at ([\d.]+)", l)
            if c:
                t0 = float(c.group(1)); break
    except FileNotFoundError:
        pass
    rows = collections.defaultdict(list)
    for l in open(fn):
        p = l.split()
        if len(p) == 5:
            rows[(p[1], p[2])].append((float(p[0]), int(p[3]), int(p[4])))
    if not rows:
        print(f"{name:12s} (no samples)"); continue
    (tid, comm), r = max(rows.items(), key=lambda kv: sum(x[2] for x in kv[1]))
    ticks = sum(x[2] for x in r)
    def share(rr):
        t = sum(x[2] for x in rr); b = sum(x[2] for x in rr if x[1] in BIG)
        return f"{100 * b / t:5.1f} ({t})" if t else "n/a"
    before = [x for x in r if t0 is None or x[0] < t0]; after = [x for x in r if t0 is not None and x[0] >= t0]
    moves = sum(1 for a, b in zip(r, r[1:]) if (a[1] in BIG) != (b[1] in BIG))
    cpus = collections.Counter()
    for x in (after if t0 else r):
        cpus[x[1]] += x[2]
    print(f"{name:12s} {comm:16s} {ticks:6d} {str(int(t0 % 1000)) if t0 else '-':>7s} {share(before):>12s} {share(after) if t0 else '-':>11s} {10 * moves / max(ticks / 100, 1):28.2f}  {dict(cpus.most_common(6))}")
