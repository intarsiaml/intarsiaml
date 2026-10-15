#!/usr/bin/env python3
"""Per-kernel table of prefill (full 2048-row chunks) and decode (mean over the 64 traced steps) for the four contexts."""
import gzip, json, os, collections, statistics as st
def load(p):
    if not os.path.exists(p) and os.path.exists(p + ".gz"): p += ".gz"
    return [json.loads(l) for l in (gzip.open(p, "rt") if p.endswith(".gz") else open(p))]
tr = {4096: "kernel/ctx4096-c2048.jsonl", 16384: "kernel/ctx16384-c2048.jsonl", 32764: "s2/kernel/ctx32768-c2048-a2.jsonl", 40000: "kernel/ctx40000-c2048.jsonl"}
def passes(rs):
    out, cur = [], None
    for r in rs:
        if r["op"] == "embedding_bf16": cur = [r]; out.append(cur)
        elif cur is not None: cur.append(r)
    return out
pre = {}; dec = {}
for ctx, p in tr.items():
    ps = passes(load(p))
    a = collections.defaultdict(list); b = collections.defaultdict(list)
    nd = 0
    for x in ps:
        if x[0]["t"] == 2048:
            for r in x: a[(r["op"], r["n"], r["k"])].append((r["e"] - r["s"]) * 1e6)
        elif x[0]["t"] == 1:
            nd += 1
            for r in x: b[(r["op"], r["n"], r["k"])].append((r["e"] - r["s"]) * 1e6)
    pre[ctx] = {k: (len(v), st.median(v)) for k, v in a.items()}
    dec[ctx] = {k: (len(v) / nd, st.median(v), len(v) / nd * st.median(v) / 1e3) for k, v in b.items()}
    npre = sum(1 for x in ps if x[0]['t'] == 2048)
    pre[ctx] = {k: (len(v) / npre, st.median(v)) for k, v in a.items()}
ctxs = list(tr)
print("Prefill, full 2048-row chunks, median us per call (kernel mode):\n")
print("| kernel | n x k | calls/chunk | " + " | ".join(str(c) for c in ctxs) + " |\n|---|---|---:|" + "---:|" * len(ctxs))
keys = sorted({k for c in ctxs for k in pre[c]}, key=lambda k: -max(pre[c].get(k, (0, 0))[1] * pre[c].get(k, (0, 0))[0] for c in ctxs))
for k in keys:
    n = max(pre[c].get(k, (0, 0))[0] for c in ctxs)
    print(f"| {k[0]} | {f'{k[1]}x{k[2]}' if k[1] else ''} | {n:.0f} | " + " | ".join(f"{pre[c][k][1]:.0f}" if k in pre[c] else "" for c in ctxs) + " |")
print("\nDecode, per generated token (kernel mode, 64 steps): median us/call and calls x median = ms/token\n")
print("| kernel | n x k | calls/token | " + " | ".join(f"{c}: us / ms" for c in ctxs) + " |\n|---|---|---:|" + "---|" * len(ctxs))
keys = sorted({k for c in ctxs for k in dec[c]}, key=lambda k: -max(dec[c].get(k, (0, 0, 0))[2] for c in ctxs))
for k in keys:
    n = max(dec[c].get(k, (0, 0, 0))[0] for c in ctxs)
    print(f"| {k[0]} | {f'{k[1]}x{k[2]}' if k[1] else ''} | {n:.0f} | " + " | ".join(f"{dec[c][k][1]:.1f} / {dec[c][k][2]:.2f}" if k in dec[c] else "" for c in ctxs) + " |")
