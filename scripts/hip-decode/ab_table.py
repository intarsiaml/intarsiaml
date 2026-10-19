#!/usr/bin/env python3
"""Interleaved A/B table from serve-bench JSONs: ab_table.py <dir> <base glob prefix> <branch glob prefix>
e.g. ab_table.py raw/bench base-pf-35b-A pf-35b-B -> per prompt size: prefill and decode tok/s of each round, the mean of
each side and the delta of the means. Only rows with errors == 0 count."""
import glob
import json
import os
import sys

d, pa, pb = sys.argv[1:4]


def load(prefix):
    runs = []
    for f in sorted(glob.glob(os.path.join(d, prefix + "*.json"))):
        rows = {r["target_prompt_tokens"]: r for r in json.load(open(f))["rows"] if r.get("errors", 0) == 0}
        runs.append((os.path.basename(f), rows))
    return runs


A, B = load(pa), load(pb)
print(f"A = {pa}* ({len(A)} runs), B = {pb}* ({len(B)} runs)")
sizes = sorted({s for _, r in A + B for s in r})
print("| prompt | A prefill tok/s | B prefill tok/s | delta | A decode tok/s | B decode tok/s | delta |")
print("|---|---|---|---|---|---|---|")
for s in sizes:
    cells = []
    for key in ("prefill_tok_s", "decode_tok_s"):
        a = [r[s][key] for _, r in A if s in r]
        b = [r[s][key] for _, r in B if s in r]
        if not a or not b:
            cells += ["-", "-", "-"]
            continue
        ma, mb = sum(a) / len(a), sum(b) / len(b)
        fmt = (lambda v: f"{v:.0f}") if key == "prefill_tok_s" else (lambda v: f"{v:.1f}")
        cells += [" / ".join(fmt(v) for v in a) + f" ({fmt(ma)})", " / ".join(fmt(v) for v in b) + f" ({fmt(mb)})", f"{(mb / ma - 1) * 100:+.2f} %"]
    print(f"| {s} | " + " | ".join(cells) + " |")
