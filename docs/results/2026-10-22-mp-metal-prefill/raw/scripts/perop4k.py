import json, sys, collections
R = sys.argv[1]
rs = [json.loads(l) for l in open(f"{R}/kernel-4k.jsonl")]
agg = collections.defaultdict(lambda: [0, 0.0, 0])
tot = 0.0
for r in rs:
    if r["t"] < 16: continue  # decode step / tiny
    d = r["e"] - r["s"]; tot += d
    k = (r["op"], r["n"], r["k"]); a = agg[k]; a[0] += 1; a[1] += d; a[2] = r["t"]
print(f"prefill kernel-mode GPU sum {tot*1e3:.1f} ms over {sum(a[0] for a in agg.values())} launches")
print(f"{'kernel':34} {'n':>6} {'k':>6} {'calls':>5} {'ms':>8} {'%':>5} {'us/call':>8} {'TFLOPS':>7}")
for k, (c, d, t) in sorted(agg.items(), key=lambda x: -x[1][1]):
    tf = ""
    if k[1] and k[2]: tf = f"{2*k[1]*k[2]*2048*c/d/1e12 if False else 0:.0f}"
    print(f"{k[0]:34} {k[1]:6} {k[2]:6} {c:5} {d*1e3:8.1f} {100*d/tot:5.1f} {d/c*1e6:8.0f}")
cb = [json.loads(l) for l in open(f"{R}/cb-4k.jsonl")]
busy = sum(r["e"] - r["s"] for r in cb); span = cb[-1]["e"] - cb[0]["s"]
print(f"cb mode: {len(cb)} command buffers, busy {busy*1e3:.1f} ms of span {span*1e3:.1f} ms ({100*busy/span:.1f}%)")
for a, b in zip(cb, cb[1:]): print(f"  gap {1e3*(b['s']-a['e']):.2f} ms  (nd {b['nd']}, dur {1e3*(b['e']-b['s']):.1f} ms)")
