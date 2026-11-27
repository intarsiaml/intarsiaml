#!/usr/bin/env python3
"""Lane ink-metal-dflash2-ready: ABBA of dfl5 vs dspark on abba3.json. abba.py DIR: DIR/abba-<arm>@<rep>.err (generate; `decode[<name>]: N tokens, ms (x tok/s)`,
`ids[<name>]: ...`); prompt 'warm' skipped. Ratio per prompt = geometric mean over reps of dfl5 / dspark; ids must be identical across all processes."""
import math, re, sys
from pathlib import Path
d = Path(sys.argv[1]); dec = re.compile(r"^decode\[([^\]]*)\]: (\d+) tokens, (\d+) ms \(([\d.]+) tok/s\)"); idl = re.compile(r"^ids\[([^\]]*)\]: (.*)$")
def rd(p):
    t, i = {}, {}
    for l in p.read_text(errors="replace").splitlines():
        if m := dec.match(l): t[m.group(1)] = float(m.group(4))
        if m := idl.match(l): i[m.group(1)] = m.group(2)
    return t, i
runs = {p.stem[5:]: rd(p) for p in sorted(d.glob("abba-*.err"))}
names = [n for n in next(iter(runs.values()))[0] if n != "warm"]
ids_same = all(runs[k][1].get(n) == next(iter(runs.values()))[1].get(n) for k in runs for n in names)
print("processes", list(runs), "| ids identical across all:", ids_same)
reps = sorted({k.split("@")[1] for k in runs}); allr = []
for n in names:
    rs = [runs[f"dfl5@{r}"][0][n] / runs[f"dspark@{r}"][0][n] for r in reps if f"dfl5@{r}" in runs and f"dspark@{r}" in runs]
    g = math.exp(sum(map(math.log, rs)) / len(rs)); allr.append(g)
    print(f"{n:24s} dspark {[runs[f'dspark@{r}'][0][n] for r in reps]} dfl5 {[runs[f'dfl5@{r}'][0][n] for r in reps]} ratio per rep {[round(x,3) for x in rs]} geo {g:.3f}")
print(f"geometric mean of the {len(allr)} prompts: {math.exp(sum(map(math.log, allr))/len(allr)):.3f}; min prompt {min(allr):.3f}")
