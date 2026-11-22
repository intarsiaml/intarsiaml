"""Host encode vs GPU time per token-feedback decode step from INTARSIA_METAL_PROFILE=cb traces (lane ink-dfuse, item 2b).
A step starts at its held-cut buffer (nd == 16). Per step: GPU busy (sum e - s), host encode (sum over its buffers of
commit-entered - first-dispatch-encoded, hc - ho), dispatches, and lead = GPU end of the step - commit of its last buffer
(> 0: the host had the step committed before the GPU finished it, so encoding is hidden).
usage: stepenc.py TRACE [TRACE...]   (prints one line per trace)"""
import json, statistics as st, sys

for path in sys.argv[1:]:
    recs = sorted((json.loads(l) for l in open(path) if l.strip()), key=lambda r: r["s"])
    starts = [i for i, r in enumerate(recs) if r["nd"] == 16]
    rows = []
    for a, b in zip(starts, starts[1:]):
        rs = recs[a:b]
        busy = sum(r["e"] - r["s"] for r in rs) * 1e3
        enc = sum(r["hc"] - r["ho"] for r in rs) * 1e3
        nd = sum(r["nd"] for r in rs)
        lead = (rs[-1]["e"] - max(r["hc"] for r in rs)) * 1e3
        rows.append((busy, enc, nd, lead))
    rows = rows[1:-1]
    m = lambda i: st.median(x[i] for x in rows)
    print(f"{path.split('/')[-1]}: steps {len(rows)} dispatches/step {m(2):.0f}  GPU busy {m(0):.3f} ms  host encode {m(1):.3f} ms "
          f"({m(1) / m(2) * 1e3:.2f} us/dispatch)  lead (GPU end - last commit) {m(3):.3f} ms")
