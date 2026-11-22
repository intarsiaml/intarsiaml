"""Per-op decode profile from an INTARSIA_METAL_PROFILE=kernel trace. usage: prof.py TRACE
Steps are split at each embedding launch; the first segment is the prefill, the first decode step is dropped (warm-up).
Logits split into unembed / router by n."""
import sys, json, collections
recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
recs.sort(key=lambda r: r["s"])
starts = [i for i, r in enumerate(recs) if r["op"].startswith("embedding")]
segs = [recs[a:b] for a, b in zip(starts, starts[1:] + [len(recs)])]
dec = segs[2:]  # drop prefill and the first decode step
steps = len(dec)
agg = collections.defaultdict(lambda: [0.0, 0, 0])
wall = []
for seg in dec:
    wall.append((seg[-1]["e"] - seg[0]["s"]) * 1e6)
    for r in seg:
        op = r["op"]
        if op.startswith("logits"):
            op += " (unembed)" if r["n"] > 100000 else " (router)"
        a = agg[op]; a[0] += (r["e"] - r["s"]) * 1e6; a[1] += 1; a[2] += r["nd"]
tot = sum(a[0] for a in agg.values()); n = sum(a[1] for a in agg.values())
print(f"decode steps {steps}: {n/steps:.1f} launches/token, {sum(a[2] for a in agg.values())/steps:.1f} dispatches/token, "
      f"{tot/steps:.0f} us summed GPU time/token (kernel mode: one command buffer per launch, incl. its fixed cost)")
for op, a in sorted(agg.items(), key=lambda kv: -kv[1][0]):
    print(f"| {op} | {a[0]/steps:.1f} | {100*a[0]/tot:.1f} | {a[1]/steps:.1f} | {a[2]/steps:.1f} | {a[0]/max(a[1],1):.1f} |")
