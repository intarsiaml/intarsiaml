#!/usr/bin/env python3
"""Lane ink-dflash2 k choice (pre-registered rule): over every traced DFlash2 k7 pass (all prompts of trace.json pooled), predicted
tok/s(k) = sum(min(accepted, k) + 1) / sum(draft ms + verify ms(k + 1 rows)); verify ms by rows from the DFlash2 + DSpark traces.
Also the geometric mean over prompts of the per-prompt prediction (for information). usage: kchoice.py DFL.err DSPARK.err"""
import math, sys
sys.argv = [sys.argv[0]] + sys.argv[1:]
import importlib.util
spec = importlib.util.spec_from_file_location("tr", __file__.replace("kchoice.py", "trace.py"))
src = open(__file__.replace("kchoice.py", "trace.py")).read().split("\ndfl = parse(")[0]
ns = {}
exec(src, ns)
dfl, dsp = ns["parse"](sys.argv[1]), ns["parse"](sys.argv[2])
rt = ns["rows_table"](dfl, dsp)
allp = [p for passes, _ in dfl.values() for p in passes]
print("pooled passes", len(allp))
for k in range(1, 8):
    tok = sum(min(a, k) + 1 for _, a, _, _ in allp)
    ms = sum(dm + rt[k + 1] for _, _, dm, _ in allp)
    per = []
    for passes, _ in dfl.values():
        if passes:
            per.append(1000 * sum(min(a, k) + 1 for _, a, _, _ in passes) / sum(dm + rt[k + 1] for _, _, dm, _ in passes))
    gm = math.exp(sum(map(math.log, per)) / len(per))
    print(f"k{k}: pooled {1000 * tok / ms:.1f} tok/s | geomean over prompts {gm:.1f}")
