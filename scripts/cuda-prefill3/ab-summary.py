#!/usr/bin/env python3
"""Summarise an A/B directory of scripts/cuda-decode/w-ab.sh runs (<dir>/35b-<arm>-r<round>/bench.log): prefill and decode
tok/s per arm, size and round, the median over rounds, and each arm's median change against BASE (default arm `base`).
  ab-summary.py <ab-dir> [base-arm]"""
import re
import sys
from pathlib import Path
from statistics import median

d = Path(sys.argv[1])
base = sys.argv[2] if len(sys.argv) > 2 else "base"
rows = {}  # (arm, size) -> {round: (prefill, decode)}
for log in sorted(d.glob("35b-*-r*/bench.log")):
    m = re.match(r"35b-(.+)-r(\d+)$", log.parent.name)
    arm, rnd = m.group(1), int(m.group(2))
    for line in log.read_text(errors="replace").splitlines():
        t = re.match(r"\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+\d+\s*$", line)
        if t:
            rows.setdefault((arm, int(t.group(1))), {})[rnd] = (float(t.group(4)), float(t.group(5)))
arms = sorted({a for a, _ in rows}, key=lambda a: (a != base, a))
sizes = sorted({s for _, s in rows})
for what, idx in (("prefill tok/s", 0), ("decode tok/s", 1)):
    print(f"\n{what} (r1, r2, r3 -> median; change vs {base} median, paired per round)")
    for s in sizes:
        b = rows.get((base, s), {})
        for a in arms:
            r = rows.get((a, s), {})
            vals = [r[k][idx] for k in sorted(r)]
            if not vals:
                continue
            med = median(vals)
            line = f"  {s:>6} {a:<10} " + ", ".join(f"{v:.1f}" if idx else f"{v:.0f}" for v in vals) + f" -> {med:.1f}"
            if a != base and b:
                bm = median([b[k][idx] for k in sorted(b)])
                paired = [100 * (r[k][idx] / b[k][idx] - 1) for k in sorted(r) if k in b]
                line += f"  {100 * (med / bm - 1):+.1f} % (paired " + ", ".join(f"{p:+.1f}" for p in paired) + ")"
            print(line)
