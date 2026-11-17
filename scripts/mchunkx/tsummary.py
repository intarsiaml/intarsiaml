#!/usr/bin/env python3
"""Lane mchunkx: compact per-group summary of CHUNKX logs. today = min(cur, today's plan measured as chunks) per M (m <= 8: the single launch).
Prints, per group and M with M >= 5: today us, best plan us (ratio), and the best plan that is not today's. Plain text, one line per M.
  tsummary.py run1.log [run2.log ...]"""
import re, sys, collections
files = [a for a in sys.argv[1:] if not a.startswith("--")]
runs = []
for f in files:
    d = {}
    for l in open(f, errors="replace"):
        m = re.match(r"CHUNKX group=(\S+) m=(\d+) plan=(\S+) us=([\d.]+)", l)
        if m:
            d[(m.group(1), int(m.group(2)), m.group(3))] = float(m.group(4))
    runs.append(d)
keys = sorted({k for d in runs for k in d})
mn = {k: min(d[k] for d in runs if k in d) for k in keys}
print("runs:", len(runs), "(min over runs)")
for g in sorted({k[0] for k in keys}):
    for m in range(2, 17):
        curplan = str(m) if m <= 8 else f"{(m+1)//2}+{m//2}"
        today = min(mn.get((g, m, "cur"), 1e18), mn.get((g, m, curplan), 1e18))
        cands = sorted((u, p) for (gg, mm, p), u in mn.items() if gg == g and mm == m and p not in ("cur", curplan))
        if not cands or today > 1e17:
            continue
        u, p = cands[0]
        # per-run ratio of best plan vs today in the same run (consistency)
        rr = []
        for d in runs:
            t = min(d.get((g, m, "cur"), 1e18), d.get((g, m, curplan), 1e18))
            if (g, m, p) in d and t < 1e17:
                rr.append(d[(g, m, p)] / t)
        flag = "  <== gain>=5%" if u / today <= 0.95 and all(r <= 0.97 for r in rr) else ""
        print(f"{g:16s} M={m:2d} today({curplan}) {today:8.0f}  best {p:9s} {u:8.0f}  ratio {u/today:.3f}  per-run {' '.join(f'{r:.3f}' for r in rr)}{flag}")

# noise floor: `cur` (the m-row call through launch) and the explicit plan that dispatches exactly the same launches, per run
if "--noise" in sys.argv:
    print("\nnoise floor: cur vs the identical explicit plan, per run (worst |ratio - 1| by group)")
    for g in sorted({k[0] for k in keys}):
        w, cnt, allr = 0.0, 0, []
        for d in runs:
            for m in range(2, 17):
                curplan = str(m) if m <= 8 else f"{(m+1)//2}+{m//2}"
                if (g, m, "cur") in d and (g, m, curplan) in d:
                    r = d[(g, m, curplan)] / d[(g, m, "cur")]
                    allr.append(abs(r - 1))
        allr.sort()
        if allr:
            print(f"{g:16s} n={len(allr):3d} median {allr[len(allr)//2]*100:4.1f}%  p90 {allr[int(len(allr)*0.9)]*100:4.1f}%  max {allr[-1]*100:4.1f}%")
