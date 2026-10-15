#!/usr/bin/env python3
"""usage: first_divergence.py <a.err> <b.err> <label>: compare the `ids: [...]` lines (greedy token ids) of two runs."""
import re
import sys


def ids(path):
    for line in open(path):
        if line.startswith("ids:"):
            return [int(x) for x in re.findall(r"-?\d+", line)]
    return None


a, b, label = ids(sys.argv[1]), ids(sys.argv[2]), sys.argv[3]
if a is None or b is None:
    print(f"{label}: no ids line ({'a' if a is None else 'b'} missing)")
    sys.exit(0)
n = min(len(a), len(b))
d = next((i for i in range(n) if a[i] != b[i]), None)
if d is None and len(a) == len(b):
    print(f"{label}: ids identical ({len(a)} tokens)")
else:
    d = n if d is None else d
    print(f"{label}: ids DIFFER, first divergence at token {d} of {len(a)}/{len(b)}")
