"""Ceiling of the ConvMix row-kernel lever: per call site, (stream row-kernel median - cached bench row-kernel median) x calls
per token, summed. Stream: a per-dispatch trace (disp.py's input); bench: ink_ctrl_bench --probe output (T 1). A call site
is told by its neighbours: plain calls by the op before (wo matmul = attention side, experts = MLP side), blend calls by the
op after the call's last stage (ink_route = attention side, else MLP side). usage: ceil.py TRACE BENCH_MD"""
import sys, json, collections, statistics as st, re
recs = sorted((json.loads(l) for l in open(sys.argv[1]) if l.strip()), key=lambda r: r["s"])
starts = [i for i, r in enumerate(recs) if r["op"].startswith("embedding")]
segs = [recs[a:b] for a, b in zip(starts, starts[1:])][2:]
n = len(segs)
site = collections.defaultdict(list)
for s in segs:
    for k, r in enumerate(s):
        if r["op"] != "conv_mix_bf16":
            continue
        d = (r["e"] - r["s"]) * 1e6
        j = k + 1
        while j < len(s) and s[j]["op"].startswith("conv_mix_bf16#"):
            j += 1
        blend = j > k + 1
        nxt = s[j]["op"] if j < len(s) else ""
        prv = s[k - 1]["op"] if k else ""
        if blend:
            key = "attn side, blend S=1" if nxt.startswith("ink_route") else "mlp side, blend S=3"
        else:
            key = "attn side, plain" if prv.startswith("matmul") else "mlp side, plain (Q 3)"
        site[key].append(d)
bench = collections.defaultdict(dict)
for l in open(sys.argv[2]):
    c = [x.strip() for x in l.strip().strip("|").split("|")]
    if len(c) >= 8 and c[1] == "1" and re.match(r"^[0-9.]+$", c[3] or ""):
        bench[c[0]][int(c[2])] = float(c[3])
tot = 0.0
print("| call site | calls / token | stream row kernel (us) | bench cached | bench evict | gap stream - cached (us) | ms / token |")
print("|---|---:|---:|---:|---:|---:|---:|")
for key in ["attn side, plain", "attn side, blend S=1", "mlp side, plain (Q 3)", "mlp side, blend S=3"]:
    v = site.get(key, [])
    if not v:
        continue
    cnt, med = len(v) / n, st.median(v)
    b = bench.get(key, {})
    cached = b.get(0)
    ev = max((x for m, x in b.items() if m > 0), default=None)
    gap = max(0.0, med - cached) if cached is not None else float("nan")
    tot += cnt * gap / 1e3
    print(f"| {key} | {cnt:.1f} | {med:.2f} | {cached} | {ev} | {gap:.2f} | {cnt * gap / 1e3:.3f} |")
import math
verdict = "INCOMPLETE (a bench number is missing)" if math.isnan(tot) else ("GO to step 4" if tot >= 0.3 else "STOP (under 0.3 ms)")
print(f"CEILING (row kernel, stream - cached, all calls): {tot:.3f} ms per token -> {verdict}")
