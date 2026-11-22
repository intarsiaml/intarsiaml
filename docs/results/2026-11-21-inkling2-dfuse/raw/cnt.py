"""Per-token op calls / dispatches from a kernel-mode trace (INTARSIA_METAL_PROFILE=kernel:F, one command buffer per op
call): steps split at each embedding launch, the prefill (first) and last partial step dropped. Prints calls per step by
(kernel, dispatches of the call) and total dispatches per step. usage: cnt.py TRACE [LABEL]"""
import sys, json, collections, statistics as st
recs = sorted((json.loads(l) for l in open(sys.argv[1]) if l.strip()), key=lambda r: r["s"])
starts = [i for i, r in enumerate(recs) if r["op"].startswith("embedding")]
segs = [recs[a:b] for a, b in zip(starts, starts[1:])][1:]
lab = sys.argv[2] if len(sys.argv) > 2 else ""
if not segs:
    print(f"COUNT {lab}: no complete steps"); sys.exit()
c = collections.Counter()
for s in segs:
    for r in s:
        c[(r["op"], r["nd"])] += 1
n = len(segs)
print(f"COUNT {lab}: {n} steps, {st.median(len(s) for s in segs):.0f} op calls and {st.median(sum(r['nd'] for r in s) for s in segs):.0f} dispatches per step")
for (op, nd), v in sorted(c.items()):
    if any(k in op for k in ("conv_mix", "loop_blend", "attn_res_mix", "moe_route", "ink_route", "logits")):
        print(f"COUNT {lab}:   {op} nd={nd}: {v / n:.1f} per step")
