#!/usr/bin/env python3
"""Lane wire-probe: the change-vs-base table from the accuracy window's raw logs.
Usage: table.py <raw dir> [prefix]   (prefix "self" = short golden, "selfl" = long golden)
Prints, per wire: prefill mean KL21, KL / reference noise, top-1 flips, decode mean KL21 and its ratio, and the verdict line."""
import re
import sys
from pathlib import Path

d = Path(sys.argv[1])
prefix = sys.argv[2] if len(sys.argv) > 2 else "self"
wires = ["off", "whtf", "wht16", "wht12", "wht8", "wht6"]
print("wire | prefill KL21 | KL / ref noise | flips (pct) | decode KL21 | decode / ref noise | prompts over own limit")
for w in wires:
    f = d / f"{prefix}-{w}.out"
    if not f.exists():
        continue
    t = f.read_text(errors="replace")
    stats = {m.group(1): m for m in re.finditer(r"SELF-STATS mode=(\w+) positions=(\d+) mean_kl21=(\S+) max_kl21=(\S+) flips=(\d+) flip_pct=(\S+)", t)}
    ratios = re.findall(r"KL / ref noise = ([\d.]+)", t)
    pre, dec = stats.get("prefill"), stats.get("decode")
    over = re.findall(r"FAIL \[([^\]]+)\] KL [^;]*needs prompt factor ([\d.]+)", t)
    print(
        f"{w} | {float(pre.group(3)):.3e} | {ratios[0] if ratios else '-'} | {pre.group(5)} ({pre.group(6)} %) | "
        f"{float(dec.group(3)):.3e} | {ratios[1] if len(ratios) > 1 else '-'} | "
        f"{', '.join(f'{a} {b}' for a, b in over) or 'none'}"
        if pre and dec
        else f"{w} | (no SELF-STATS)"
    )
