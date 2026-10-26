#!/usr/bin/env python3
"""Lane sk: aggregate localeval speed results of one le-window.sh run (raw/<out>/index.txt) per arm:
median of the per-round prefill tok/s medians (and min-max over rounds), TTFT, marginal prefill, decode.
usage: agg.py <raw/outdir> [baseline-arm]   (arm = name in sk-<arm>-r<N>)"""
import json, re, statistics as st, sys, os, collections
d = sys.argv[1]; base = sys.argv[2] if len(sys.argv) > 2 else None
runs = collections.defaultdict(list)
for line in open(os.path.join(d, "index.txt")):
    m = re.match(r"sk-(.+)-r(\d+): .*results/([^/]+)/result.json", line)
    if m: runs[m.group(1)].append((int(m.group(2)), m.group(3)))
cells = {}
for arm, rs in runs.items():
    for _, rid in rs:
        j = json.load(open(f"/opt/code/deepeval/results/{rid}/result.json"))
        for row in j["speed"]["rows"]:
            c = cells.setdefault((arm, row["target_prompt_tokens"]), collections.defaultdict(list))
            c["tok"].append(row["prefill_tok_s"]["median"]); c["ttft"].append(row["ttft_s"]["median"])
            c["dec"].append(row["decode_tok_s"]["median"]); c["ptok"].append(row["prompt_tokens"])
            if row.get("marginal_prefill_tok_s"): c["marg"].append(row["marginal_prefill_tok_s"])
sizes = sorted({k[1] for k in cells}); arms = sorted(runs)
print("| arm | " + " | ".join(f"prefill {s//1024}k tok/s (min-max) [prompt tok]" for s in sizes) + " | " + " | ".join(f"TTFT {s//1024}k s" for s in sizes) + " | decode 1k/4k/16k |")
print("|---|" + "---|" * (2 * len(sizes) + 1))
for a in arms:
    t = [cells[(a, s)] for s in sizes]
    print(f"| {a} | " + " | ".join(f"{st.median(c['tok']):.0f} ({min(c['tok']):.0f}-{max(c['tok']):.0f}) [{c['ptok'][0]}]" for c in t) + " | " +
          " | ".join(f"{st.median(c['ttft']):.3f}" for c in t) + " | " + " / ".join(f"{st.median(c['dec']):.1f}" for c in t) + " |")
if base:
    for a in arms:
        if a == base: continue
        print(f"{a} vs {base}: " + ", ".join(f"{s//1024}k {100*(st.median(cells[(a,s)]['tok'])/st.median(cells[(base,s)]['tok'])-1):+.1f} %" for s in sizes))
