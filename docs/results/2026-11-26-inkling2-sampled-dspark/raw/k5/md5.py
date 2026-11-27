#!/usr/bin/env python3
"""Lane ink-sampled greedy md5 table: `md5.py DIR [names...]` reads DIR/md5-<tree>-<spec>.err (`ids[<prompt>]: ...` from `generate
--show-ids`), trees main / br*, spec off / def, and prints per prompt the md5 of the first 64 ids and of all 256, and whether every
file agrees."""
import hashlib
import re
import sys
from pathlib import Path

d = Path(sys.argv[1])
idl = re.compile(r"^ids\[([^\]]*)\]: (.*)$")
runs = {}
for f in sorted(d.glob("md5-*.err")):
    ids = {}
    for line in f.read_text(errors="replace").splitlines():
        m = idl.match(line)
        if m:
            ids[m.group(1)] = m.group(2).split()
    runs[f.stem[4:]] = ids
names = list(next(iter(runs.values())).keys())
h = lambda v: hashlib.md5(" ".join(v).encode()).hexdigest()[:8]
print("| prompt | " + " | ".join(f"{r} @64 / @256" for r in runs) + " | all equal |")
print("|---|" + "---|" * len(runs) + "---|")
allok = True
for n in names:
    cells = [f"{h(runs[r][n][:64])} / {h(runs[r][n][:256])} ({len(runs[r][n])})" for r in runs]
    ok = len({tuple(runs[r][n]) for r in runs}) == 1
    allok &= ok
    print(f"| {n} | " + " | ".join(cells) + f" | {'yes' if ok else 'NO'} |")
print(f"\nall prompts identical across {list(runs)}: {'yes' if allok else 'NO'}")
