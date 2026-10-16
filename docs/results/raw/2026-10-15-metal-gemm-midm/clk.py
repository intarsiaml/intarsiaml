#!/usr/bin/env python3
"""GPU clock of the busy samples (util >= 0.5) of macmon pipe logs: python3 clk.py file..."""
import json, sys
for f in sys.argv[1:]:
    fr = []
    for l in open(f):
        try:
            j = json.loads(l)
        except Exception:
            continue
        if j["gpu_usage"][1] >= 0.5:
            fr.append(j["gpu_usage"][0])
    fr.sort()
    if fr:
        print(f"{f}: busy samples {len(fr)}, GPU MHz median {fr[len(fr)//2]}, p10 {fr[len(fr)//10]}, p90 {fr[9*len(fr)//10]}")
    else:
        print(f"{f}: no busy samples")
