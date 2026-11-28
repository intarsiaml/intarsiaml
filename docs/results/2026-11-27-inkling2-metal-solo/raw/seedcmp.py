#!/usr/bin/env python3
"""Lane ink-metal-dflash2-ready: sampled-id identity between arms. seedcmp.py DIR REF ARM...: DIR holds s-<arm>-T<T>-k<K>.err (generate --show-ids:
`ids[<prompt>]: ...`); every (config, prompt) pair of ARM is compared with REF's. Prints per config: pairs compared, pairs differing."""
import re, sys
from pathlib import Path
d, ref, arms = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
idl = re.compile(r"^ids\[([^\]]*)\]: (.*)$")
def rd(p):
    return {m.group(1): m.group(2) for l in p.read_text(errors="replace").splitlines() if (m := idl.match(l))}
cfgs = sorted({p.name[len(f"s-{ref}-"):-4] for p in d.glob(f"s-{ref}-*.err")})
tot = {a: [0, 0] for a in arms}
for c in cfgs:
    r = rd(d / f"s-{ref}-{c}.err")
    for a in arms:
        x = rd(d / f"s-{a}-{c}.err"); common = sorted(set(r) & set(x)); bad = [k for k in common if r[k] != x[k]]
        tot[a][0] += len(common); tot[a][1] += len(bad)
        print(f"{c:12s} {a:6s} vs {ref}: ref {len(r)} / arm {len(x)} prompts, compared {len(common)}, differ {len(bad)} {bad[:3]}")
for a in arms: print(f"TOTAL {a} vs {ref}: compared {tot[a][0]}, differ {tot[a][1]}")
