"""Decode-step timeline of a real run: the union of token times over the streams (one batched step = one token per decoding stream),
with the first-token (pass end) marks."""
import json, os, sys
out, visit, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
tmax = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
f = [x for x in os.listdir(out) if x.startswith(f"{visit}-real") and x.endswith(f"-n{n}.json") and "meta" not in x][0]
d = json.load(open(os.path.join(out, f))); meta = json.load(open(os.path.join(out, f.replace(".json", ".meta.json")))); T0 = meta["t0"]
ss = sorted(d.values(), key=lambda r: r["ttft"])
marks = {}
for i, r in enumerate(ss):
    marks.setdefault(round(r["t0"] + r["ttft"] - T0, 2), []).append(f"first p{r['prompt_tokens']}")
times = []
for r in ss:
    t = r["t0"] + r["ttft"]
    for g in r["gaps"]:
        t += g; times.append(t - T0)
times.sort()
# cluster token times within 5 ms = one step
steps = []
for t in times:
    if steps and t - steps[-1][0] < 0.012: steps[-1][1] += 1
    else: steps.append([t, 1])
prev = None
for t, c in steps:
    if t > tmax: break
    m = [f"{k}:{','.join(v)}" for k, v in marks.items() if prev is not None and prev < k <= t + 0.001]
    print(f"{t:7.3f} +{(t - prev) if prev else 0:6.3f} rows {c:2d} {' '.join(m)}")
    prev = t
