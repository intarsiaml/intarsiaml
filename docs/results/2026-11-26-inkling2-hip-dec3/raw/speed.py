#!/usr/bin/env python3
"""Paired decode gain of the branch over main from a window log's SPD lines (stage_speed.sh): per prompt, the ratio branch / main
of each rep (the two arms of a rep ran back to back, ABBA over the reps), its mean and a t-based 95 % CI over the reps; and the
md5 of every run. usage: speed.py window.log"""
import collections, math, re, sys
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
tok = collections.defaultdict(dict)
md5 = collections.defaultdict(set)
for line in open(sys.argv[1], errors="replace"):
    m = re.search(r"SPD (\w+) (\w+) (\d+) r(\d+) ([0-9a-f]{32}) \| decode \d+ ms \(([0-9.]+) tok/s\)", line)
    if m:
        arm, p, n, rep, h, v = m.groups()
        tok[(p, n)][(arm, int(rep))] = float(v)
        md5[(p, n)].add((arm, h))
for (p, n), d in sorted(tok.items()):
    reps = sorted({r for (a, r) in d if a == "main"} & {r for (a, r) in d if a == "branch"})
    q = [d[("branch", r)] / d[("main", r)] for r in reps]
    mn = [d[("main", r)] for r in reps]
    br = [d[("branch", r)] for r in reps]
    k = len(q)
    mean = sum(q) / k
    sd = math.sqrt(sum((x - mean) ** 2 for x in q) / (k - 1)) if k > 1 else float("nan")
    h = T95.get(k - 1, 2.0) * sd / math.sqrt(k) if k > 1 else float("nan")
    print(f"{p:5s} {n:>3s} tokens: main {min(mn):.2f}-{max(mn):.2f} (mean {sum(mn)/k:.2f}), branch {min(br):.2f}-{max(br):.2f} (mean {sum(br)/k:.2f}) tok/s; "
          f"paired ratio {mean:.4f} (+{100*(mean-1):.2f} %), 95 % CI [{mean-h:.4f}, {mean+h:.4f}], n={k}")
    print("   md5:", ", ".join(f"{a}={h[:8]}" for a, h in sorted(md5[(p, n)])))
