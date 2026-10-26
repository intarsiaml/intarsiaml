#!/usr/bin/env python3
"""Lane dfh: acceptance per draft position from INTARSIA_SPEC_TRACE logs (`spec: token N pos P drafted D accepted A emits E`).
usage: python3 -I tools/dfh/accept_table.py <raw dir with gen-<tp>-<prompt>-<arm>.err files>
Per (tp, arm, prompt length class): passes, tokens per pass (A + 1), and P(accepted >= i) for i = 1..7 (the chance a pass
gets past draft i)."""
import collections, glob, os, re, sys
raw = sys.argv[1]
pat = re.compile(r"spec: token \d+ pos \d+ drafted (\d+) accepted (\d+) emits")
agg = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(raw, "gen-*-*-*.err"))):
    m = re.match(r"gen-(tp\d)-(short|4k|16k)-\d+-(\w+)\.err", os.path.basename(f))
    if not m or m.group(3) == "off":
        continue
    tp, kind, arm = m.groups()
    for line in open(f, errors="replace"):
        mm = pat.search(line)
        if mm:
            agg[(tp, arm, kind)].append((int(mm.group(1)), int(mm.group(2))))
print(f"{'tp':4} {'arm':5} {'prompts':7} {'passes':>6} {'t/pass':>6}  P(accepted >= i), i = 1..7")
for (tp, arm, kind), v in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1], ["short", "4k", "16k"].index(kv[0][2]))):
    n = len(v)
    tpp = sum(a + 1 for _, a in v) / n
    kmax = max(d for d, _ in v)
    ps = " ".join(f"{sum(1 for _, a in v if a >= i) / n:.2f}" for i in range(1, kmax + 1))
    print(f"{tp:4} {arm:5} {kind:7} {n:6} {tpp:6.2f}  {ps}")
