"""Decode step period from `generate` cb traces (INTARSIA_METAL_PROFILE=cb:, --profile-every 0): a normal step ends with
its largest command buffer; period = end-to-end distance of consecutive step ends, GPU busy = sum of the buffers in
between. usage: step_period.py TRACE... (prints one line per trace, then the median over all of them)"""
import sys, json, statistics as st
allp = []
for path in sys.argv[1:]:
    recs = sorted((json.loads(l) for l in open(path) if l.strip()), key=lambda r: r["s"])
    big = max(r["nd"] for r in recs)
    idx = [i for i, r in enumerate(recs) if r["nd"] >= big - 2]
    per, busy = [], []
    for a, b in zip(idx, idx[1:]):
        if b - a > 6:
            continue
        per.append((recs[b]["e"] - recs[a]["e"]) * 1e6)
        busy.append(sum(r["e"] - r["s"] for r in recs[a + 1:b + 1]) * 1e6)
    per, busy = per[5:], busy[5:]  # skip the first steps (clock ramp)
    allp += per
    print(f"{path}: {len(per)} steps, period median {st.median(per):.0f} us (p10 {sorted(per)[len(per)//10]:.0f}, p90 {sorted(per)[9*len(per)//10]:.0f}), GPU busy {st.median(busy):.0f} us -> {1e6/st.median(per):.1f} tok/s")
if len(sys.argv) > 2:
    print(f"all: median period {st.median(allp):.0f} us -> {1e6/st.median(allp):.2f} tok/s")
