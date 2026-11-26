"""Solo gate summary (lane ink-hip-batch): md5 equality per (prompt, tokens) and the branch / main decode ratio per prompt (paired by
rep, 95 % CI with Student t), from the SPD lines of stage_gate.sh.  python3 gate_ratio.py LOG"""
import collections
import re
import statistics
import sys

T95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 8: 2.365}
md5 = collections.defaultdict(set)
dec = collections.defaultdict(dict)
pre = collections.defaultdict(dict)
for line in open(sys.argv[1], errors="replace"):
    m = re.search(r"SPD (\w+) (\w+) (\d+) r(\d+) ([0-9a-f]{32}) \| decode \d+ ms \(([0-9.]+) tok/s\) \| prefill (\d+) ms", line)
    if not m:
        continue
    arm, p, n, r, h, d, pf = m.groups()
    md5[(p, n)].add((arm, h))
    if n == "256":
        dec[(p, arm)][r] = float(d)
        pre[(p, arm)][r] = float(pf)
for (p, n), s in sorted(md5.items()):
    hs = {h for _, h in s}
    print(f"md5 {p} {n}: {'EQUAL' if len(hs) == 1 else 'DIFFER'} ({', '.join(sorted(h[:12] for h in hs))})")
for what, tab in (("decode tok/s", dec), ("prefill ms", pre)):
    for p in ("short", "4k"):
        a, b = tab.get((p, "main"), {}), tab.get((p, "branch"), {})
        reps = sorted(set(a) & set(b))
        if not reps:
            continue
        rs = [b[r] / a[r] for r in reps]
        if what == "prefill ms":
            rs = [1 / x for x in rs]  # as a speed ratio
        mu = statistics.mean(rs)
        ci = T95.get(len(rs), 2.0) * statistics.stdev(rs) / len(rs) ** 0.5 if len(rs) > 1 else 0
        print(f"{what} {p}: main {' / '.join(f'{a[r]:g}' for r in reps)}; branch {' / '.join(f'{b[r]:g}' for r in reps)}; "
              f"ratio {mu:.4f} +- {ci:.4f}")
