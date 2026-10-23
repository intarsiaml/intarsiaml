#!/usr/bin/env python3
"""Lane pf: tabulate a scripts/pf/le-window.sh window from its index.txt (label -> localeval result.json).
   le-table.py <raw/w1/index.txt> [--ref vllm]    prefill tok/s per size: per round and median, delta vs the reference arm; decode median."""
import json, re, sys, statistics as st, collections, os
idx = sys.argv[1]
ref = "vllm"
if "--ref" in sys.argv: ref = sys.argv[sys.argv.index("--ref") + 1]
rows = collections.defaultdict(dict)  # arm -> round -> {size: (prefill, decode, marginal)}
for line in open(idx):
    m = re.match(r"pf-(.+)-r(\d+): .*results/([^/`]+)/result.json", line)
    if not m: continue
    arm, r, run = m.group(1), int(m.group(2)), m.group(3)
    f = f"/opt/code/deepeval/results/{run}/result.json"
    if not os.path.exists(f): continue
    d = json.load(open(f))["speed"]["rows"]
    rows[arm][r] = {x["target_prompt_tokens"]: (x["prefill_tok_s"]["median"], x["decode_tok_s"]["median"], x.get("marginal_prefill_tok_s")) for x in d}
sizes = [1024, 4096, 16384]
arms = list(rows)
med = {a: {s: st.median([rows[a][r][s][0] for r in rows[a] if s in rows[a][r]]) for s in sizes} for a in arms}
dec = {a: {s: st.median([rows[a][r][s][1] for r in rows[a] if s in rows[a][r]]) for s in sizes} for a in arms}
print("| arm | rounds | prefill tok/s 1k | 4k | 16k | vs %s 1k | 4k | 16k | decode tok/s 1k / 4k / 16k |" % ref)
print("|---|---|---|---|---|---|---|---|---|")
for a in arms:
    per = {s: "/".join(f"{rows[a][r][s][0]:.0f}" for r in sorted(rows[a]) if s in rows[a][r]) for s in sizes}
    dl = {s: (f"{100*(med[a][s]/med[ref][s]-1):+.1f}%" if ref in med and a != ref else "ref") for s in sizes}
    print(f"| {a} | {len(rows[a])} | {med[a][1024]:.0f} ({per[1024]}) | {med[a][4096]:.0f} ({per[4096]}) | {med[a][16384]:.0f} ({per[16384]}) | {dl[1024]} | {dl[4096]} | {dl[16384]} | {dec[a][1024]:.1f} / {dec[a][4096]:.1f} / {dec[a][16384]:.1f} |")
