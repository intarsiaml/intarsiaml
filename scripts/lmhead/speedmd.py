#!/usr/bin/env python3
"""Markdown speed table from hdrive.sh / mspeed.sh steps.log files: usage speedmd.py LABEL steps.log [steps.log...]
Rows = (config, size); columns = bf16-head median decode tok/s (both visits), quantised-head median decode, ratio. Config = item head minus the quantiser."""
import json, re, sys, glob, collections
LE = "/opt/code/deepeval/"
def load(steps):
    runs = collections.defaultdict(list)   # (cfg, q, size) -> [(decode_median, visit_index)]
    for line in open(steps):
        m = re.search(r"item (\d+) (\S+) speed: exit 0 (results/\S+)", line)
        if not m: continue
        n, full, res = m.groups(); parts = full.split(".")
        if len(parts) == 5: b, q, tp, spec, mb = parts; cfg = f"{b} tp{tp} {'MTP k2' if spec == 'mtp' else 'plain'}"
        else: q, spec = parts; cfg = f"metal {'MTP k2' if spec == 'mtp' else 'plain'}"
        r = json.load(open(LE + res))
        for row in r["speed"]["rows"]:
            runs[(cfg, q, row["target_prompt_tokens"])].append((row["decode_tok_s"]["median"], int(n)))
    return runs
print("| config | prompt | bf16 head tok/s (A, A) | quantised | head tok/s (B, B) | B/A (medians of the two visits) |\n|---|---|---|---|---|---|")
for steps in sys.argv[2:]:
    runs = load(steps); cfgs = sorted({k[0] for k in runs}); 
    for cfg in cfgs:
        for q in sorted({k[1] for k in runs if k[0] == cfg and k[1] != "off"}):
            for size in sorted({k[2] for k in runs if k[0] == cfg}):
                a = [v for v, _ in sorted(runs.get((cfg, "off", size), []), key=lambda t: t[1])]; b = [v for v, _ in sorted(runs.get((cfg, q, size), []), key=lambda t: t[1])]
                if not a or not b: continue
                med = lambda xs: sorted(xs)[len(xs)//2] if len(xs) % 2 else sum(sorted(xs)[len(xs)//2-1:len(xs)//2+1]) / 2
                print(f"| {cfg} | {size} | " + ", ".join(f"{x:.1f}" for x in a) + f" | {q} | " + ", ".join(f"{x:.1f}" for x in b) + f" | {med(b)/med(a):.3f} |")
