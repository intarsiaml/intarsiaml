#!/usr/bin/env python3
"""Paired gain of the branch over main from window logs' SPD lines (stage_speed.sh): per setting (spec-prompt), the ratio branch / main
of each rep (the two arms of a rep ran back to back, ABBA over the reps), its mean and a t-based 95 % CI over the reps, for decode
and prefill tok/s; and the md5 of every run. usage: speed.py window.log [window.log ...]"""
import collections, math, re, sys
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
tok = {"decode": collections.defaultdict(dict), "prefill": collections.defaultdict(dict)}
md5 = collections.defaultdict(set)
for path in sys.argv[1:]:
    for line in open(path, errors="replace"):
        m = re.search(r"SPD (\w+) ([\w-]+) (\d+) r(\d+) ([0-9a-f]{32}) \| (.*)", line)
        if not m:
            continue
        arm, p, n, rep, h, rest = m.groups()
        md5[(p, n)].add((arm, h))
        for kind in ("decode", "prefill"):
            v = re.search(kind + r" \d+ ms \(([0-9.]+) tok/s\)", rest)
            if v:
                tok[kind][(p, n)][(arm, int(rep))] = float(v.group(1))


def stats(d):
    reps = sorted({r for (a, r) in d if a == "main"} & {r for (a, r) in d if a == "branch"})
    if not reps:
        return None
    q = [d[("branch", r)] / d[("main", r)] for r in reps]
    mn = [d[("main", r)] for r in reps]
    br = [d[("branch", r)] for r in reps]
    k = len(q)
    mean = sum(q) / k
    sd = math.sqrt(sum((x - mean) ** 2 for x in q) / (k - 1)) if k > 1 else float("nan")
    h = T95.get(k - 1, 2.0) * sd / math.sqrt(k) if k > 1 else float("nan")
    return sum(mn) / k, sum(br) / k, mean, mean - h, mean + h, k


print("| setting | metric | main tok/s | branch tok/s | ratio | 95 % CI | n |")
print("|---|---|---|---|---|---|---|")
for kind in ("decode", "prefill"):
    for (p, n), d in sorted(tok[kind].items()):
        s = stats(d)
        if s:
            mn, br, mean, lo, hi, k = s
            print(f"| {p} ({n} tok) | {kind} | {mn:.2f} | {br:.2f} | {mean:.4f} ({100*(mean-1):+.2f} %) | [{lo:.4f}, {hi:.4f}] | {k} |")
print()
for (p, n), s in sorted(md5.items()):
    print(f"md5 {p} {n}: " + ", ".join(f"{a}={h[:8]}" for a, h in sorted(s)))
