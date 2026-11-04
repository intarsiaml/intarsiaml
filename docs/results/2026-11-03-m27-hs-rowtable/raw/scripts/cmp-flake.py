#!/usr/bin/env python3
"""ids of flake-hunt runs vs the def-mb2 reference (same prompt set, greedy): per run, how many of the (rounds x 12) requests differ."""
import json, sys, glob
ref = json.load(open("def-mb2-ids.json"))
refq = {k.split("-")[1]: v["ids"] for k, v in ref.items()}
for f in sorted(sys.argv[1:] or glob.glob("flake*-ids.json")):
    d = json.load(open(f))
    bad = [k for k, v in d.items() if v["ids"] != refq[k.split("-")[1]]]
    print(f"{f}: {len(d) - len(bad)}/{len(d)} requests identical to def-mb2 ids" + (f"  DIFF {bad}" if bad else ""))
