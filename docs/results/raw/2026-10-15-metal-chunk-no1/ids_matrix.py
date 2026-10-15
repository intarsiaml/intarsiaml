#!/usr/bin/env python3
"""Compare the greedy ids of every run of a (L, prompt) group against the base single-chunk run."""
import re, sys, glob
def ids(p):
    for l in open(p):
        if l.startswith("ids:"): return [int(x) for x in re.findall(r"-?\d+", l)]
for L in (257, 2049, 4097):
    for s in (1, 2):
        ref = ids(f"identity/base-L{L}-s{s}-c{L}.err")
        row = []
        for b in ("base", "after"):
            for c in (256, 2048, L):
                try: a = ids(f"identity/{b}-L{L}-s{s}-c{c}.err")
                except FileNotFoundError: row.append(f"{b}@{c}=missing"); continue
                if a == ref: row.append(f"{b}@{c}=same")
                else:
                    d = next((i for i in range(min(len(a), len(ref))) if a[i] != ref[i]), min(len(a), len(ref)))
                    row.append(f"{b}@{c}=DIFF@{d}")
        print(f"L={L} s={s}: " + "  ".join(row))
