"""Controller time per token from a `generate --profile-every 16` cb trace (Fable's method, raw/cb_steps.py): normal steps give
wall / GPU busy / dispatches; profiled steps (one waited command buffer per op call) give per-op GPU time, corrected by the
per-command-buffer fixed cost c = (profiled GPU - normal busy) / buffers, so the corrected ops sum to the normal step.
usage: ctrl.py TRACE"""
import sys, json, collections, statistics as st
recs = sorted((json.loads(l) for l in open(sys.argv[1]) if l.strip()), key=lambda r: r["s"])
big = collections.Counter(r["nd"] for r in recs if r["nd"] > 100).most_common(1)[0][0]
idx = [i for i, r in enumerate(recs) if r["nd"] == big]
norm, prof = [], []
for a, b in zip(idx, idx[1:]):
    if b - a <= 6:
        seg = recs[a + 1:b + 1]
        norm.append(((recs[b]["e"] - recs[a]["e"]) * 1e6, sum(r["e"] - r["s"] for r in seg) * 1e6, sum(r["nd"] for r in seg)))
    else:
        prof.append(recs[a + 1:b - 1])
wall = st.median(x[0] for x in norm); busy = st.median(x[1] for x in norm); nd = st.median(x[2] for x in norm)
CTRL = ("conv_mix", "loop_blend", "attn_res_mix", "moe_route", "ink_route", "router_gemv")
agg = collections.defaultdict(list)
ncb = []
for s in prof:
    ncb.append(len(s))
    for r in s:
        op, d = r["op"], (r["e"] - r["s"]) * 1e6
        if op.startswith("logits"): op = "unembed" if d > 500 else "router_gemv"
        if op.startswith(("conv_mix", "loop_blend", "moe_experts")): op += f" nd{r['nd']}"
        agg[op].append(d)
n = len(prof); tot = sum(sum(v) for v in agg.values()) / n
c = (tot - busy) / st.mean(ncb)
print(f"normal steps {len(norm)}: wall {wall:.0f} us ({1e6 / wall:.2f} tok/s), GPU busy {busy:.0f} us, dispatches {nd:.0f} (held cut + rest)")
print(f"profiled steps {n}: {st.mean(ncb):.0f} op calls, GPU {tot:.0f} us; per-buffer fixed cost c = {c:.2f} us")
ctl_raw = ctl_cor = 0.0
for op, v in sorted(agg.items(), key=lambda kv: -sum(kv[1])):
    per = sum(v) / n; cnt = len(v) / n; cor = per - c * cnt
    is_c = op.startswith(CTRL)
    if is_c: ctl_raw += per; ctl_cor += cor
    print(f"| {op} | {per:.0f} | {cor:.0f} | {cnt:.0f} | {st.median(v):.1f} | {'ctrl' if is_c else ''} |")
print(f"CONTROLLERS per token: {ctl_raw / 1e3:.3f} ms raw, {ctl_cor / 1e3:.3f} ms corrected (c = {c:.2f} us per call)")
