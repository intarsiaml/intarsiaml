#!/usr/bin/env python3
"""s1sum.py <tag> [copy]: per-arm localeval table (decode/prefill by size), % of 40.6 and of the context-aware floor, clock.
With 'copy', copies each localeval result.json into raw/<tag>/localeval-results/."""
import glob, json, os, shutil, statistics as st, subprocess, sys

tag = sys.argv[1]
D = os.path.dirname(os.path.abspath(__file__))
O = os.path.join(D, tag)
BW, W, KVB = 614e9, 15.1368e9, 147456
def ctxfloor(p): return BW / (W + KVB * (p + 128))
for f in sorted(glob.glob(f"/opt/code/deepeval/results/*-adhoc-g3-{tag}-*/result.json")):
    d = json.load(open(f))
    lab = d["label"].replace(f"g3-{tag}-", "")
    if len(sys.argv) > 2:
        dst = os.path.join(O, "localeval-results", os.path.basename(os.path.dirname(f)))
        os.makedirs(dst, exist_ok=True); shutil.copy(f, dst)
    cells = []
    for r in d["speed"]["rows"]:
        dec = r["decode_tok_s"]; p = r["prompt_tokens"]
        cells.append(f"{p}: {dec['median']:.1f} ({dec['min']:.1f}-{dec['max']:.1f}) {100*dec['median']/40.6:.1f}%/{100*dec['median']/ctxfloor(p):.1f}%cf pf {r['prefill_tok_s']['median']:.0f}")
    mm = subprocess.run([sys.executable, os.path.join(D, "mm.py"), os.path.join(O, f"{lab}-macmon.jsonl")], capture_output=True, text=True).stdout.strip()
    print(f"## {lab} [{d['run_id']}]\n  " + "\n  ".join(cells) + f"\n  {mm}")
