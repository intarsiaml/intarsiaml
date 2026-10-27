#!/usr/bin/env python3
"""Lane cgv: paired per-round decode comparison of le-window.sh arms (raw/<out>/index.txt, localeval result.json files).
    pairs.py <raw/outdir> new:base [new:base ...]
Per size and pair: the per-round decode tok/s medians of both arms, the paired ratios new/base, their median and min-max, and the spread of each arm over rounds
((max - min) / median). Markdown on stdout."""
import collections, json, os, re, statistics as st, sys
d = sys.argv[1]; pairs = [p.split(":") for p in sys.argv[2:]]
runs = collections.defaultdict(dict)
for line in open(os.path.join(d, "index.txt")):
    m = re.match(r"cgv-(.+)-r(\d+): .*results/([^/]+)/result.json", line)
    if m: runs[m.group(1)][int(m.group(2))] = m.group(3)
cell = {}
for arm, rs in runs.items():
    for r, rid in rs.items():
        j = json.load(open(f"/opt/code/deepeval/results/{rid}/result.json"))
        for row in j["speed"]["rows"]:
            cell[(arm, row["target_prompt_tokens"], r)] = (row["decode_tok_s"]["median"], row["prompt_tokens"])
sizes = sorted({k[1] for k in cell})
def vals(a, s): return [cell[(a, s, r)][0] for r in sorted(runs[a]) if (a, s, r) in cell]
print("| size | arm | n | per-round decode tok/s | median | spread |")
print("|---|---|---:|---|---:|---:|")
for s in sizes:
    for a in sorted(runs):
        v = vals(a, s)
        if v: print(f"| {s} | {a} | {len(v)} | {', '.join(f'{x:.2f}' for x in v)} | {st.median(v):.2f} | {100*(max(v)-min(v))/st.median(v):.1f} % |")
print()
print("| size | pair (new / base) | n | paired ratios by round | median | min-max | ratio of medians |")
print("|---|---|---:|---|---:|---|---:|")
for s in sizes:
    for new, base in pairs:
        rr = sorted(set(r for (a, ss, r) in cell if a == new and ss == s) & set(r for (a, ss, r) in cell if a == base and ss == s))
        q = [cell[(new, s, r)][0] / cell[(base, s, r)][0] for r in rr]
        if q: print(f"| {s} | {new} / {base} | {len(q)} | {', '.join(f'{x:.3f}' for x in q)} | {st.median(q):.3f} | {min(q):.3f}-{max(q):.3f} | {st.median(vals(new,s))/st.median(vals(base,s)):.3f} |")
