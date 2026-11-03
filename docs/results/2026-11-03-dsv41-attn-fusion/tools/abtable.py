#!/usr/bin/env python3
"""Per-config prefill walls from run-full.sh output directories (ab0, ab1, ...), from each steps.log line
`pf <tag> <prompt file> wall <s> s {usage...}` (tags plainN-rep / longN-rep; warm-ups ignored).

Usage: abtable.py DIR [LABEL_FILE]   (DIR holds ab0/, ab1/, ...; LABEL_FILE: one label per config, in order, optional)
Prints per config and prompt: n, wall seconds of each rep, median, tok/s of the median (prompt_tokens / wall).
"""
import re, statistics as st, sys, os, glob
d = sys.argv[1]
labels = open(sys.argv[2]).read().splitlines() if len(sys.argv) > 2 else []
pat = re.compile(r"pf (plain|long)\d+-\d+ (\S+) wall ([\d.]+) s .*?'prompt_tokens': (\d+)")
for i, a in enumerate(sorted(glob.glob(os.path.join(d, "ab*")), key=lambda p: int(re.search(r"ab(\d+)$", p).group(1)))):
    rows = {}
    for l in open(os.path.join(a, "steps.log")):
        m = pat.search(l)
        if m:
            rows.setdefault(m[2], ([], int(m[4])))[0].append(float(m[3]))
    lab = labels[i] if i < len(labels) else os.path.basename(a)
    for f, (w, n) in rows.items():
        md = st.median(w)
        print(f"{lab:<44} {f:<14} n={len(w)} walls {' '.join(f'{x:.3f}' for x in w)}  median {md:.3f} s  {n / md:7.1f} tok/s ({n} tokens)")
