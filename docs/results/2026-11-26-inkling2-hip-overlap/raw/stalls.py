"""Line up multi-stream stalls in a real-prompt run (client gaps) with arrivals, prefill spans, finishes and the server poll."""
import json, os, sys, collections, re
out, visit, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
thr = float(sys.argv[4]) if len(sys.argv) > 4 else 0.3
files = [f for f in os.listdir(out) if f.startswith(f"{visit}-real") and f.endswith(f"-n{n}.json") and "meta" not in f]
d = json.load(open(os.path.join(out, files[0])))
meta = json.load(open(os.path.join(out, files[0].replace(".json", ".meta.json"))))
T0 = meta["t0"]
streams = []
for k, r in d.items():
    t = r["t0"] + r["ttft"]
    toks = [t]
    for g in r["gaps"]:
        t += g; toks.append(t)
    streams.append(dict(name=k, t0=r["t0"], first=r["t0"] + r["ttft"], t1=r["t1"], ptok=r["prompt_tokens"], toks=toks, gaps=r["gaps"]))
streams.sort(key=lambda s: s["t0"])
print(f"{visit} n={n} wall {meta['wall_s']} s; t relative to the run's t0")
for i, s in enumerate(streams):
    s["i"] = i
    print(f"  s{i} {s['name']:10s} arrive {s['t0']-T0:6.3f} first {s['first']-T0:6.3f} (ttft {s['first']-s['t0']:.3f}) end {s['t1']-T0:6.3f} prompt {s['ptok']}")
# events: gaps > thr
ev = []
for s in streams:
    for j, g in enumerate(s["gaps"]):
        if g > thr:
            ev.append((s["toks"][j] - T0, s["toks"][j + 1] - T0, s["i"], g))
ev.sort()
clusters = []
for e in ev:
    if clusters and e[0] < clusters[-1]["end"]:
        c = clusters[-1]; c["m"].append(e); c["end"] = max(c["end"], e[1]); c["start"] = min(c["start"], e[0])
    else:
        clusters.append(dict(start=e[0], end=e[1], m=[e]))
print(f"gaps > {thr}: {len(ev)} in {len(clusters)} events")
for c in clusters:
    st, en = c["start"], c["end"]
    pre = [s["i"] for s in streams if s["t0"] - T0 <= en and s["first"] - T0 >= st]  # prefilling during the event
    arr = [(s["i"], round(s["t0"] - T0, 3)) for s in streams if st - 1.0 <= s["t0"] - T0 <= en]
    firsts = [(s["i"], round(s["first"] - T0, 3)) for s in streams if st - 0.05 <= s["first"] - T0 <= en + 0.05]
    dec = [s["i"] for s in streams if s["first"] - T0 < st and s["t1"] - T0 > en]
    print(f"  event {st:6.3f}-{en:6.3f}: {len(c['m'])} streams {[ (m[2], round(m[3],3), round(m[0],3), round(m[1],3)) for m in c['m']]}")
    print(f"     decoding through it {dec}; prefilling during it {pre}; first tokens near {firsts}; arrivals in [st-1, en] {arr}")
# all gaps > 0.15 per stream timeline counts
print("gap histogram (all streams):", collections.Counter(min(int(g / 0.05), 12) * 50 for s in streams for g in s["gaps"]))
