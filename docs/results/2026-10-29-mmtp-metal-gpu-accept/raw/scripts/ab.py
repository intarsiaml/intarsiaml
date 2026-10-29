#!/usr/bin/env python3
"""MMTP: A/B table from localeval result.json files of the served rounds. ab.py DIR"""
import json, glob, os, re, sys, statistics as S, collections
d = sys.argv[1]
cells = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(f"{d}/r*-localeval.out")):
    tag = os.path.basename(f)[:-len("-localeval.out")]
    r, arm = tag.split("-", 1)
    m = re.search(r"saved (\S+result.json)", open(f).read() + open(f.replace(".out", ".err")).read())
    if not m: print("no result", f); continue
    j = json.load(open(m.group(1)))
    txt = open(f).read()
    for line in txt.splitlines():
        c = [x.strip() for x in line.split("|")]
        if len(c) > 9 and c[1].isdigit():
            p = int(c[1]); size = min([1024, 4096, 16384], key=lambda s: abs(s - p))
            cells[(arm, size)]["decode"].append(float(c[5].split()[0]))
            cells[(arm, size)]["prefill"].append(float(c[3].split()[0]))
            cells[(arm, size)]["ttft"].append(float(c[2]))
arms = sorted({a for a, _ in cells})
print("| arm | size | decode tok/s (per round) | median | prefill tok/s (per round) | median |")
print("|---|---:|---|---:|---|---:|")
for a in arms:
    for s in [1024, 4096, 16384]:
        c = cells.get((a, s))
        if not c: continue
        print(f"| {a} | {s} | {' / '.join(f'{x:.1f}' for x in c['decode'])} | {S.median(c['decode']):.1f} | {' / '.join(f'{x:.0f}' for x in c['prefill'])} | {S.median(c['prefill']):.0f} |")
def rel(a, b, s, k):
    return S.median(cells[(a, s)][k]) / S.median(cells[(b, s)][k]) - 1
for a, b in [("mtp1", "mtp0"), ("off-new", "off-main")]:
    if (a, 4096) in cells and (b, 4096) in cells:
        print(f"{a} vs {b}: " + ", ".join(f"decode {s}: {rel(a,b,s,'decode')*100:+.1f}%" for s in [1024, 4096, 16384]) + f", prefill 4k: {rel(a,b,4096,'prefill')*100:+.1f}%")
