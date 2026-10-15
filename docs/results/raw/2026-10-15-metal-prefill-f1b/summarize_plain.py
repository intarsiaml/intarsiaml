#!/usr/bin/env python3
"""usage: summarize_plain.py <plain dir> <label A> <label B>
Reads A-<prompt>-c<chunk>-<round><a|b|c|d>.err (plain-abba.sh output; A = runs a and d, B = runs b and c) and prints per
config the prefill ms of every run, the mean and median per binary, and B against A."""
import glob, re, statistics, sys, collections
d, la, lb = sys.argv[1], sys.argv[2], sys.argv[3]
runs = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(f"{d}/[AB]-*-c*-*.err")):
    m = re.match(r".*/([AB])-(\w+)-c(\d+)-(\d+)([abcd])\.err", f)
    if not m:
        continue
    who, prompt, chunk, rnd, pos = m.groups()
    t = re.findall(r"prefill (\d+) ms", open(f).read())
    if t:
        runs[(prompt, int(chunk))][who].append(int(t[-1]))
print(f"| prompt | chunk | {la} ms (n, mean, median) | {lb} ms (n, mean, median) | {lb} vs {la} (median) | {lb} vs {la} (mean) |")
print("|---|---:|---:|---:|---:|---:|")
for (prompt, chunk), v in sorted(runs.items(), key=lambda kv: (kv[0][0] != "4k", kv[0][1])):
    a, b = v["A"], v["B"]
    ma, mb = statistics.median(a), statistics.median(b)
    print(f"| {prompt} | {chunk} | {len(a)}, {statistics.mean(a):.0f}, {ma:.0f} | {len(b)}, {statistics.mean(b):.0f}, {mb:.0f} | {100*(mb/ma-1):+.1f}% | {100*(statistics.mean(b)/statistics.mean(a)-1):+.1f}% |")
print()
for (prompt, chunk), v in sorted(runs.items(), key=lambda kv: (kv[0][0] != "4k", kv[0][1])):
    print(f"{prompt} c{chunk}: {la} {v['A']}  {lb} {v['B']}")
