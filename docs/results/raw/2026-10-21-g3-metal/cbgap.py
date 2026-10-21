#!/usr/bin/env python3
"""cbgap.py <cb-trace.jsonl[.gz]> [steps]: decode step period / GPU busy / idle gap, and the gap split into
  wake  = previous step's last buffer GPU end -> host starts encoding the next step's first buffer (ho)
  enc   = ho -> commit entered (hc)       (host encode of the first buffer)
  launch= commit returned (hr) -> GPU start of that buffer (s)
over the last <steps> decode steps (a step ends at the buffer with the argmax; record fields per profile.rs)."""
import gzip, json, sys, statistics as st

f = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 128
op = gzip.open if f.endswith(".gz") else open
recs = [json.loads(l) for l in op(f, "rt")]
# decode buffers: those of the steps after prefill. A step = buffers up to and including one whose label is argmax
steps, cur = [], []
for r in recs:
    cur.append(r)
    if r.get("op", "").startswith("argmax"):
        steps.append(cur); cur = []
steps = steps[-n:]
period, busy, gap, wake, enc, launch, disp = [], [], [], [], [], [], []
for a, b in zip(steps, steps[1:]):
    first = b[0]
    period.append((b[0]["s"] - a[0]["s"]) * 1e3)
    busy.append(sum(r["e"] - r["s"] for r in a) * 1e3)
    gap.append((first["s"] - a[-1]["e"]) * 1e3)
    if "ho" in first:
        wake.append((first["ho"] - a[-1]["e"]) * 1e3)
        enc.append((first["hc"] - first["ho"]) * 1e3)
        launch.append((first["s"] - first["hr"]) * 1e3)
    disp.append(sum(r["nd"] for r in a))
md = lambda x: st.median(x) if x else float("nan")
print(f"{f}: {len(period)} step pairs, dispatches/step {md(disp):.0f}, buffers/step {md([len(s) for s in steps]):.0f}")
print(f"period {md(period):.3f} ms ({1e3/md(period):.2f} tok/s)  busy {md(busy):.3f}  idle gap {md(gap):.3f} ms (p10 {st.quantiles(gap, n=10)[0]:.3f} p90 {st.quantiles(gap, n=10)[-1]:.3f})")
if wake:
    print(f"gap split (median): wake {md(wake):.3f}  encode-first-buffer {md(enc):.3f}  commit->GPU start {md(launch):.3f} ms")
