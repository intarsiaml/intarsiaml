"""Per-op GPU us per call from a per-dispatch cb trace (INTARSIA_TOKEN_FEEDBACK=0 INTARSIA_METAL_COMMIT_EVERY=1
INTARSIA_METAL_FIRST_COMMIT=0, --profile-every 0): one labelled command buffer per launch, not waited on. Decode steps
start at an `embedding*` record; the last STEPS steps are aggregated. usage: cb1_ops.py TRACE [STEPS=24]"""
import sys, json, collections, statistics as st
recs = sorted((json.loads(l) for l in open(sys.argv[1]) if l.strip()), key=lambda r: r["s"])
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 24
starts = [i for i, r in enumerate(recs) if r["op"].startswith("embedding")]
starts = starts[-(steps + 1):]
agg = collections.defaultdict(list)
spans, busy = [], []
for a, b in zip(starts, starts[1:]):
    w = recs[a:b]
    spans.append((recs[b]["s"] - recs[a]["s"]) * 1e6)
    busy.append(sum(r["e"] - r["s"] for r in w) * 1e6)
    for r in w:
        op = r["op"]
        if op.startswith("logits"):
            op += " (unembed)" if r["n"] > 100000 else " (router)"
        agg[op].append((r["e"] - r["s"]) * 1e6)
n = len(spans)
print(f"{n} steps: span median {st.median(spans):.0f} us, GPU busy median {st.median(busy):.0f} us per token")
print("| op | us/token | calls/token | median us/call | p10 | p90 |")
for op, v in sorted(agg.items(), key=lambda kv: -sum(kv[1])):
    s = sorted(v)
    print(f"| {op} | {sum(v)/n:.0f} | {len(v)/n:.0f} | {st.median(v):.1f} | {s[len(s)//10]:.1f} | {s[9*len(s)//10]:.1f} |")
