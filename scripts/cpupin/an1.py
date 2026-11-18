#!/usr/bin/env python3
"""cpupin: tables of the core-placement A/B (scripts/cpupin/ab.sh output dir DIR: base, baseP, n20, n20P x rounds): per server the engine thread's modal cpu, the mean graph step and
the median decode tok/s of its requests; per arm the round medians and the ratios n20/base, n20P/baseP, baseP/base. usage: an1.py DIR"""
import re, sys, glob, json, collections, statistics as st
d = sys.argv[1]
G = re.compile(r"^spec: mtp graph step ([\d.]+) ms gap ([\d.]+) ms n (\d+) len (\d+)")
rows = {}
for fn in sorted(glob.glob(f"{d}/serve-*-r*.log")):
    name = fn.split("serve-")[1][:-4]
    c = collections.Counter()
    for l in open(f"{d}/ts-{name}.tsv"):
        p = l.split()
        if p[2] == "intarsia-engine":
            c[int(p[3])] += int(p[4])
    steps = [float(G.match(l).group(1)) for l in open(fn, errors="replace") if G.match(l)]
    j = [json.loads(l) for l in open(f"{d}/{name}.jsonl") if l.startswith("{")]
    rows[name] = (c.most_common(1)[0][0], len(steps), st.mean(steps), st.median(x["dec_tps"] for x in j))
print("server    cpu  steps  mean_graph_step_ms  median_dec_tps")
for n, r in sorted(rows.items(), key=lambda kv: (kv[0].split("-")[0], kv[0])):
    print(f"{n:9s} {r[0]:3d} {r[1]:6d} {r[2]:18.3f} {r[3]:14.1f}")
arms = ["base", "baseP", "n20", "n20P"]
rounds = sorted({int(n.split("-r")[1]) for n in rows})
print("\nround medians of decode tok/s (code-gen 800 tokens, 8 reps)")
for a in arms:
    v = [rows[f"{a}-r{r}"][3] for r in rounds]
    print(f"{a:6s} {' '.join(f'{x:6.1f}' for x in v)}   mean {st.mean(v):.2f}")
for num, den in [("n20", "base"), ("n20P", "baseP"), ("baseP", "base"), ("n20P", "base")]:
    v = [rows[f"{num}-r{r}"][3] / rows[f"{den}-r{r}"][3] for r in rounds]
    s = [rows[f"{num}-r{r}"][2] - rows[f"{den}-r{r}"][2] for r in rounds]
    print(f"{num}/{den}: tok/s ratio per round {' '.join(f'{x:.4f}' for x in v)}  mean {st.mean(v):.4f}; mean graph step difference per round (ms) {' '.join(f'{x:+.3f}' for x in s)}  mean {st.mean(s):+.3f}")
