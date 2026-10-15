#!/usr/bin/env python3
"""analyze_json.py <events.log> <macmon.jsonl> <label-prefix> ...: per pass (localeval result.json, label cbm0-gap-<label>): decode / prefill
tok/s medians per prompt size, and the GPU clock (macmon, median of the samples with GPU util >= 50% inside the pass window)."""
import json, glob, sys, statistics as st
from datetime import datetime
ev, mmf = sys.argv[1], sys.argv[2]
win = {}
for l in open(ev):
    p = l.split()
    if len(p) == 3: win.setdefault(p[0], {})[p[1]] = float(p[2])
mm = []
for line in open(mmf):
    try: d = json.loads(line)
    except Exception: continue
    mm.append((datetime.fromisoformat(d["timestamp"]).timestamp(), d["gpu_usage"][0], d["gpu_usage"][1]))
rows = []
for label, w in win.items():
    fs = glob.glob(f"/opt/code/deepeval/results/*-adhoc-cbm0-gap-{label}/result.json")
    if not fs: continue
    r = json.load(open(fs[0]))["speed"]["rows"]
    busy = [s[1] for s in mm if w["start"] < s[0] < w["end"] and s[2] >= 0.5]
    rows.append((w["start"], label, {x["target_prompt_tokens"]: x for x in r}, st.median(busy) if busy else float("nan"), json.load(open(fs[0]))["run_id"]))
rows.sort()
print("| pass | decode 128 | decode 1k | decode 4k | prefill 128 / 1k / 4k (tok/s) | GPU MHz (util>=50%) | localeval run |")
print("|---|---|---|---|---|---|---|")
for _, label, r, mhz, rid in rows:
    d = lambda s: f"{r[s]['decode_tok_s']['median']:.2f} ({r[s]['decode_tok_s']['min']:.1f}-{r[s]['decode_tok_s']['max']:.1f})"
    pf = " / ".join(f"{r[s]['prefill_tok_s']['median']:.0f}" for s in (128, 1024, 4096))
    print(f"| {label} | {d(128)} | {d(1024)} | {d(4096)} | {pf} | {mhz:.0f} | {rid} |")
