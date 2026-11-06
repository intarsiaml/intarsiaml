#!/usr/bin/env python3
"""Lane bm: speed table from a raw dir of localeval runs (<label>.out naming the result.json, <label>.metrics with the spec counters).
table.py <rawdir>: one row per (arm, streams): aggregate decode tok/s per round, mean, per-stream decode median, TTFT median, run ids."""
import glob, json, os, re, sys
from collections import defaultdict
raw = sys.argv[1]
rows = defaultdict(dict)
acc = defaultdict(list)
for out in sorted(glob.glob(os.path.join(raw, "bm-*-r*.out"))):
    label = os.path.basename(out)[:-4]
    m = re.match(r"bm-(?:metal-)?(.+)-r(\d+)$", label)
    arm, r = m.group(1), int(m.group(2))
    txt = open(out, errors="replace").read()
    p = re.search(r"saved (\S+result\.json)", txt)
    if not p:
        continue
    d = json.load(open(p.group(1)))
    rid = os.path.basename(os.path.dirname(p.group(1)))
    for c in d["speed"]["concurrency"]:
        rows[(arm, c["streams"])][r] = (c["aggregate_decode_tok_s"], c["per_stream_decode_tok_s"]["median"], c["ttft_s"]["median"], rid)
    mf = os.path.join(raw, label + ".metrics")
    if os.path.exists(mf):
        mt = dict(l.split()[:2] for l in open(mf) if l.strip() and not l.startswith("#"))
        prop, a = float(mt.get("intarsia_spec_proposed_tokens_total", 0)), float(mt.get("intarsia_spec_accepted_tokens_total", 0))
        if prop:
            acc[arm].append(a / prop)
print("| arm | streams | aggregate r1 / r2 | mean | per-stream (median, r1 / r2) | TTFT p50 s (r1 / r2) |")
print("|---|---|---|---|---|---|")
for (arm, s), rr in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    ks = sorted(rr)
    ag = [rr[k][0] for k in ks]
    print(f"| {arm} | {s} | {' / '.join(f'{x:.1f}' for x in ag)} | {sum(ag)/len(ag):.1f} | {' / '.join(f'{rr[k][1]:.1f}' for k in ks)} | {' / '.join(f'{rr[k][2]:.2f}' for k in ks)} |")
print()
for arm, v in sorted(acc.items()):
    print(f"acceptance {arm}: " + " / ".join(f"{x*100:.1f}%" for x in v))
print()
for (arm, s), rr in sorted(rows.items()):
    if s == 1:
        print(f"run ids {arm}: " + ", ".join(rr[k][3] for k in sorted(rr)))
