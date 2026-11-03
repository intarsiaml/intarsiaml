#!/usr/bin/env python3
"""cb-mode profile (INTARSIA_METAL_PROFILE=cb:FILE) of a decode: per step GPU busy time, wall time and dispatches.
A decode step = a 4-dispatch command buffer (the token feedback) followed by 128-dispatch buffers and a short last one.
usage: cbstats.py FILE [steps]"""
import json, statistics as st, sys
recs = [json.loads(l) for l in open(sys.argv[1])]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 30
starts = [i for i in range(1, len(recs)) if recs[i]["nd"] == 4 and recs[i - 1]["nd"] == 48]
starts = starts[-(n + 1):]
busy, wall, disp, ncb = [], [], [], []
for a, b in zip(starts, starts[1:]):
    s = recs[a:b]
    busy.append(sum(r["e"] - r["s"] for r in s))
    wall.append(recs[b]["s"] - recs[a]["s"])
    disp.append(sum(r["nd"] for r in s))
    ncb.append(len(s))
print(f"{len(busy)} steps: {disp[0]} dispatches and {ncb[0]} command buffers per step, GPU busy {st.median(busy)*1e3:.2f} ms, wall {st.median(wall)*1e3:.2f} ms, GPU idle {(st.median(wall)-st.median(busy))*1e3:.2f} ms ({(1-st.median(busy)/st.median(wall))*100:.1f} %)")
