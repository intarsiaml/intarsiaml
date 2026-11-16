#!/usr/bin/env python3
"""kvq: summarize ab-*/visits.jsonl. Decode tok/s is read from the saved localeval tables (column 'decode tok/s'),
server ms/token from /metrics. Prints per cell: arm means, spread, on/off, flagged visits."""
import json, glob, re, statistics as st, sys, os
raw = sys.argv[1]
def toks(path):
    hdr = None
    for line in open(path):
        if line.startswith("|"):
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if hdr is None and "decode tok/s" in c: hdr = c; continue
            if hdr and c[0].replace(",", "").isdigit():
                try: return float(c[hdr.index("decode tok/s")].split()[0])
                except Exception: return None
for d in sorted(glob.glob(raw + "/ab-*/")):
    rec = [json.loads(l) for l in open(d + "visits.jsonl")]
    print("##", os.path.basename(d.rstrip("/")))
    for size in sorted({r["size"] for r in rec}):
        res = {}
        for arm in "AB":
            rs = [r for r in rec if r["arm"] == arm and r["size"] == size and not r["rerun"]]
            v = []
            for r in rs:
                f = f"{d}{r['idx']:02d}-{arm}/localeval-{size}.out"
                v.append((toks(f), r["srv_ms_per_token"], r["srv_ms_per_pass"]))
            res[arm] = v
        A = [x[0] for x in res["A"] if x[0]]; B = [x[0] for x in res["B"] if x[0]]
        ma, mb = st.mean(A), st.mean(B)
        sp = lambda x: (max(x) - min(x)) / st.mean(x) * 100
        ms = lambda arm: st.mean(x[1] for x in res[arm])
        pp = [x[2] for a in "AB" for x in res[a] if x[2]]
        print(f"size {size}: off {ma:.1f} (n={len(A)}, spread {sp(A):.1f}%)  on {mb:.1f} (n={len(B)}, spread {sp(B):.1f}%)  on/off {mb/ma:.3f}  "
              f"srv ms/tok off {ms('A'):.2f} on {ms('B'):.2f}" + (f"  ms/pass off {st.mean(x[2] for x in res['A']):.1f} on {st.mean(x[2] for x in res['B']):.1f}" if pp else ""))
        print("   off:", [round(x,1) for x in A], " on:", [round(x,1) for x in B])
    print("reruns:", [(r['idx'], r['arm'], r['size']) for r in rec if r['rerun']])
