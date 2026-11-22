"""Per-dispatch decode timeline (INTARSIA_METAL_COMMIT_EVERY=1, INTARSIA_TOKEN_FEEDBACK=0, --profile-every 0: one pipelined,
unwaited command buffer per dispatch). Steps split at each embedding launch; the prefill and the first decode step dropped.
Per op label: GPU us per token, dispatches per token, median us per dispatch; the controllers summed. usage: disp.py TRACE"""
import sys, json, collections, statistics as st
recs = sorted((json.loads(l) for l in open(sys.argv[1]) if l.strip()), key=lambda r: r["s"])
starts = [i for i, r in enumerate(recs) if r["op"].startswith("embedding")]
segs = [recs[a:b] for a, b in zip(starts, starts[1:])][2:]
CTRL = ("conv_mix", "loop_blend", "attn_res_mix", "moe_route", "ink_route", "router_gemv")
agg = collections.defaultdict(list)
busy, span, nds = [], [], []
for s in segs:
    busy.append(sum(r["e"] - r["s"] for r in s) * 1e6)
    span.append((s[-1]["e"] - s[0]["s"]) * 1e6)
    nds.append(sum(r["nd"] for r in s))
    for r in s:
        op, d = r["op"], (r["e"] - r["s"]) * 1e6
        if op.startswith("logits"): op = "unembed" if r.get("n", 0) > 100000 else "router_gemv"
        agg[op].append(d)
n = len(segs)
print(f"{n} decode steps: GPU busy {st.median(busy):.0f} us, span {st.median(span):.0f} us, dispatches {st.median(nds):.0f}")
ctl = 0.0
for op, v in sorted(agg.items(), key=lambda kv: -sum(kv[1])):
    per = sum(v) / n
    if op.startswith(CTRL): ctl += per
    print(f"| {op} | {per:.0f} | {len(v) / n:.0f} | {st.median(v):.2f} | {'ctrl' if op.startswith(CTRL) else ''} |")
print(f"CONTROLLERS (per-dispatch, pipelined): {ctl / 1e3:.3f} ms per token")
