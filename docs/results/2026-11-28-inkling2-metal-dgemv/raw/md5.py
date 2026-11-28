#!/usr/bin/env python3
"""Lane ink-metal-dflash2-ready: md5 of the `ids[<prompt>]: [...]` lines of generate --show-ids (the `[a, b, ...]` string as printed, #546's
convention) per arm: md5.py g-off.err g-default.err g-dfl5.err ..."""
import hashlib, re, sys
idl = re.compile(r"^ids\[([^\]]*)\]: (.*)$")
runs = {}
for f in sys.argv[1:]:
    ids = {}
    for l in open(f, errors="replace"):
        m = idl.match(l.rstrip("\n"))
        if m: ids[m.group(1)] = m.group(2)
    runs[f.rsplit("/", 1)[-1]] = ids
names = list(next(iter(runs.values())))
print("| prompt | " + " | ".join(runs) + " | n ids | equal |"); print("|---|" + "---|" * (len(runs) + 2))
for n in names:
    vals = [runs[r].get(n, "") for r in runs]
    cnt = [len(re.findall(r"\d+", v)) for v in vals]
    print(f"| {n} | " + " | ".join(hashlib.md5(v.encode()).hexdigest() for v in vals) + f" | {cnt} | {'yes' if len(set(vals)) == 1 and vals[0] else 'NO'} |")
