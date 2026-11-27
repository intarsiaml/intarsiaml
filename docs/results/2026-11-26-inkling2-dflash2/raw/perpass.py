#!/usr/bin/env python3
"""Lane ink-dflash2: ms per verify pass (draft + verify + rollback, i.e. decode ms / passes) per class, from the ABBA logs'
decode[<name>] and spec[<name>] lines, pooled over the processes. usage: perpass.py DIR "short=1-12,..." ARM ..."""
import re, sys
from pathlib import Path
d, cls, arms = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
C = []
for part in cls.split(","):
    n, r = part.split("=")
    idx = []
    for piece in r.split("+"):
        a, _, b = piece.partition("-")
        idx += list(range(int(a), int(b or a) + 1))
    C.append((n, idx))
dec = re.compile(r"^decode\[([^\]]*)\]: (\d+) tokens, (\d+) ms")
spp = re.compile(r"^spec\[([^\]]*)\]: verify passes=(\d+) proposed=(\d+) accepted=(\d+)")
print("| ms per verify pass | " + " | ".join(n for n, _ in C) + " |")
print("|---|" + "---:|" * len(C))
for a in arms:
    ms, ps = {}, {}
    for p in d.glob(f"{a}@*.err"):
        i = -1
        for line in p.read_text(errors="replace").splitlines():
            if m := dec.match(line):
                i += 1
                ms[i] = ms.get(i, 0) + int(m.group(3))
            if m := spp.match(line):
                ps[i] = ps.get(i, 0) + int(m.group(2))
    cells = []
    for n, idx in C:
        t, q = sum(ms.get(i, 0) for i in idx), sum(ps.get(i, 0) for i in idx)
        cells.append(f"{t / q:.2f}" if q else "-")
    print(f"| {a} | " + " | ".join(cells) + " |")
