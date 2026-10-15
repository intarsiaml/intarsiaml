#!/usr/bin/env python3
"""ops_sum.py <dir>: per build and op, the per-pair us/call medians of small_ops runs, and fingerprint agreement."""
import sys, glob, os, re, collections
d = sys.argv[1]
t = collections.defaultdict(list); fp = collections.defaultdict(dict)
for f in sorted(glob.glob(os.path.join(d, "*.out"))):
    label = os.path.basename(f).rsplit("-", 1)[0]
    for line in open(f):
        m = re.match(r"^(norm_add|qkrope|silu|argmax)\s+([\d.]+)\s+[\d.]+\s+[\d.]+$", line)
        if m: t[(label, m.group(1))].append(float(m.group(2)))
        m = re.match(r"^(.{40}) ([0-9a-f]{16})$", line.rstrip("\n"))
        if m: fp[label][m.group(1).strip()] = m.group(2)
for op in ["norm_add", "qkrope", "silu", "argmax"]:
    print(op + ": " + "  ".join(f"{l} {' '.join(f'{x:.2f}' for x in v)}" for (l, o), v in sorted(t.items()) if o == op))
base = fp.get("pre", {})
for l, m in sorted(fp.items()):
    diff = [k for k in base if m.get(k) != base[k]]
    print(f"fingerprints {l}: {len(m)} cases, differ from pre: {diff if diff else 'none'}")
