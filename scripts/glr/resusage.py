#!/usr/bin/env python3
"""Lane GLR: compact table (VGPRs, scratch, LDS, waves/SIMD) from hipcc -Rpass-analysis=kernel-resource-usage output, kernels whose
mangled name contains the filter words (all given words must occur)."""
import re, subprocess, sys
txt = open(sys.argv[1]).read()
words = sys.argv[2:]
cur = None; rows = {}
for l in txt.splitlines():
    m = re.search(r"Function Name: (\S+)", l)
    if m: cur = m.group(1); rows[cur] = {}; continue
    m = re.search(r"remark: .*?(VGPRs|AGPRs|ScratchSize \[bytes/lane\]|Occupancy \[waves/SIMD\]|LDS Size \[bytes/block\]): (\d+)", l)
    if m and cur: rows[cur][m.group(1).split()[0]] = int(m.group(2))
for name, r in rows.items():
    if all(w in name for w in words):
        dn = subprocess.run(["c++filt", name], capture_output=True, text=True).stdout.strip() or name
        dn = re.sub(r"\(A8Args.*", "", dn).replace("(anonymous namespace)::", "").replace("void ", "")
        print(f"{dn} | VGPR {r.get('VGPRs')} | scratch {r.get('ScratchSize')} | LDS {r.get('LDS')} | occ {r.get('Occupancy')}")
