#!/usr/bin/env python3
"""Lane mfp8split: pooled 35B ABBA over the runs (every visit of every run, same binary, env switch): per cell n, mean / median of per-visit aggregate tok/s and step ms.
usage: pooled.py <dir> [<dir>...]"""
import json, glob, os, sys, statistics as st, re, gzip
agg, step = {}, {}
for d in sys.argv[1:]:
    for v in sorted(glob.glob(d + "/*-mb16")):
        arm = os.path.basename(v)[0]
        win = {}
        for l in open(v + "/cells.txt"):
            m = re.match(r"([\d.]+) (start|end) (\S+)", l)
            if m: win.setdefault(m.group(3), [0, 0])[m.group(2) == "end"] = float(m.group(1))
        em = []
        for l in gzip.open(v + "/server.err.gz", "rt", errors="replace"):
            m = re.match(r"cb-trace-us t=(\d+) Emit \{(.*)\}", l)
            if m:
                kv = dict(re.findall(r"(\w+): (\d+)", m.group(2)))
                em.append((int(m.group(1)) / 1e6, (int(kv.get("prepare_us", 0)) + int(kv.get("verify_us", 0)) + int(kv.get("decode_us", 0))) / 1e3))
        for f in glob.glob(v + "/s*.json"):
            tag = os.path.basename(f)[:-5]
            agg.setdefault(tag, {}).setdefault(arm, []).append(json.load(open(f))["summary"]["agg_tps"])
            t0, t1 = win[tag]
            e = [x for t, x in em if t0 <= t <= t1]
            if e: step.setdefault(tag, {}).setdefault(arm, []).append(st.median(e))
print("| cell | visits A / B | A agg tok/s mean (median) | B agg tok/s mean (median) | B / A mean | B / A median | A step ms (median of visits) | B step ms | step B / A | visits where B's step < A's median |")
print("|---|---|---|---|---|---|---|---|---|---|")
for tag in sorted(agg, key=lambda t: int(t[1:])):
    A, B = agg[tag]["A"], agg[tag]["B"]; sA, sB = step[tag]["A"], step[tag]["B"]
    ma = st.median(sA)
    print(f"| {tag} | {len(A)} / {len(B)} | {st.mean(A):.1f} ({st.median(A):.1f}) | {st.mean(B):.1f} ({st.median(B):.1f}) | {st.mean(B)/st.mean(A):.3f} | {st.median(B)/st.median(A):.3f} | "
          f"{st.median(sA):.1f} | {st.median(sB):.1f} | {st.median(sB)/st.median(sA):.3f} | {sum(1 for x in sB if x < ma)} / {len(sB)} |")
