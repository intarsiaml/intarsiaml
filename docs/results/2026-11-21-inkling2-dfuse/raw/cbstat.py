"""Decode-step timeline from an INTARSIA_METAL_PROFILE=cb trace.

usage: cbstat.py TRACE MODE [STEPS]
  MODE fb    : token-feedback decode; a step starts at each held-cut buffer (nd == HOLD, default 16)
  MODE plain : plain decode; a step starts at each first-commit buffer (nd == 4) after the prefill; the last STEPS of them
  MODE enc   : host encode cost per dispatch from full commit_every buffers (open -> commit of nd == 128 buffers)
Prints per-step wall (start to next start), GPU busy, GPU idle, and for fb the release slack (commit -> GPU start of the
held rest) and the GPU time of the held-cut buffer. Steps exclude the first and the last (warm-up, stop).
"""
import json, statistics as stt, sys

recs = sorted((json.loads(l) for l in open(sys.argv[1]) if l.strip()), key=lambda r: r["s"])
mode = sys.argv[2]
hold = 16


def med(v):
    return stt.median(v) if v else float("nan")


def mean(v):
    return stt.fmean(v) if v else float("nan")


if mode == "enc":
    per = [(r["hc"] - r["ho"]) / r["nd"] * 1e6 for r in recs if r["nd"] == 128]
    print(f"encode buffers {len(per)}: host us/dispatch median {med(per):.2f} mean {mean(per):.2f} min {min(per):.2f}")
    sys.exit()

if mode == "fb":
    starts = [i for i, r in enumerate(recs) if r["nd"] == hold]
else:
    n = int(sys.argv[3])
    starts = [i for i, r in enumerate(recs) if r["nd"] == 4][-n:]
steps = []
for a, b in zip(starts, starts[1:]):
    span = recs[b]["s"] - recs[a]["s"]
    busy = sum(r["e"] - r["s"] for r in recs[a:b])
    nd = sum(r["nd"] for r in recs[a:b])
    st = {"span": span * 1e3, "busy": busy * 1e3, "idle": (span - busy) * 1e3, "nd": nd, "ncb": b - a}
    if mode == "fb" and b - a >= 2:
        rest = recs[a + 1]
        st["slack"] = (rest["s"] - rest["hc"]) * 1e3
        st["cut_gpu"] = (recs[a]["e"] - recs[a]["s"]) * 1e3
        st["rest_gap"] = (rest["s"] - recs[a]["e"]) * 1e3
    steps.append(st)
steps = steps[1:-1]
if not steps:
    print("no steps found")
    sys.exit(1)
k = steps[0].keys()
print(f"steps {len(steps)}  dispatches/step {med([s['nd'] for s in steps]):.0f}  buffers/step {med([s['ncb'] for s in steps]):.0f}")
for key in ["span", "busy", "idle"] + (["slack", "cut_gpu", "rest_gap"] if mode == "fb" else []):
    v = [s[key] for s in steps if key in s]
    print(f"  {key:9s} ms/step median {med(v):7.3f} mean {mean(v):7.3f} min {min(v):7.3f} max {max(v):7.3f}")
tot_span = sum(s["span"] for s in steps)
tot_busy = sum(s["busy"] for s in steps)
print(f"  total: wall {tot_span:.1f} ms busy {tot_busy:.1f} ms idle {100 * (1 - tot_busy / tot_span):.2f} %")
