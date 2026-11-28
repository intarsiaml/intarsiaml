#!/usr/bin/env python3
"""Lane ink-metal-dtopk: medians of the DFlash2 draft phases (`spec: dflash2 draft flush .. block .. head .. selproj .. dl .. select .. total .. ms`,
INTARSIA_SPEC_TRACE=1) per file, and dl + select."""
import re, statistics, sys
pat = re.compile(r"spec: dflash2 draft flush ([\d.]+) block ([\d.]+) head ([\d.]+) selproj ([\d.]+) dl ([\d.]+) select ([\d.]+) total ([\d.]+) ms")
names = ["flush", "block", "head", "selproj", "dl", "select", "total"]
print("| file | passes | " + " | ".join(names) + " | dl+select |\n|---|---:|" + "---:|" * (len(names) + 1))
for f in sys.argv[1:]:
    rows = [tuple(map(float, m.groups())) for l in open(f, errors="replace") if (m := pat.search(l))]
    if not rows:
        print(f"| {f} | 0 |"); continue
    med = [statistics.median(r[i] for r in rows) for i in range(len(names))]
    ds = statistics.median(r[4] + r[5] for r in rows)
    print(f"| {f.split('/')[-1]} | {len(rows)} | " + " | ".join(f"{x:.3f}" for x in med) + f" | {ds:.3f} |")
