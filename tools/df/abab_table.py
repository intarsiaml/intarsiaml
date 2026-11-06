#!/usr/bin/env python3
"""Lane df: ABAB table from localeval suite runs labelled df-mr-<visit>-<spec>-<A|B>-<cell> (scripts/df/metal-abab.sh).
Per (spec, cell, prompt size): decode tok/s median of each visit, the A and B medians over visits, B/A.
Usage: abab_table.py [results dir] [label prefix, default df-mr-]"""
import glob, json, os, re, statistics as st, sys
from collections import defaultdict
res = sys.argv[1] if len(sys.argv) > 1 else "/opt/code/deepeval/results"
pre = sys.argv[2] if len(sys.argv) > 2 else "df-mr-"
v = defaultdict(lambda: defaultdict(list))
for f in glob.glob(os.path.join(res, f"*adhoc-{pre}*", "result.json")):
    m = re.search(re.escape(pre) + r"(\d+)-(\w+)-([AB])-(speed-\w+)", f)
    if not m:
        continue
    _, sp, arm, cell = m.groups()
    for r in json.load(open(f))["speed"]["rows"]:
        d = r.get("decode_tok_s") or {}
        med = d.get("median") if isinstance(d, dict) else d
        if med:
            v[(sp, cell, r["target_prompt_tokens"])][arm].append(med)
print("| arm | cell | prompt | A visits | B visits | A | B | B/A |")
print("|---|---|---:|---|---|---:|---:|---:|")
for (sp, cell, size), a in sorted(v.items()):
    A, B = a.get("A", []), a.get("B", [])
    if A and B:
        ma, mb = st.median(A), st.median(B)
        print(f"| {sp} | {cell} | {size} | {', '.join(f'{x:.1f}' for x in A)} | {', '.join(f'{x:.1f}' for x in B)} | {ma:.1f} | {mb:.1f} | {mb / ma:.3f} |")
