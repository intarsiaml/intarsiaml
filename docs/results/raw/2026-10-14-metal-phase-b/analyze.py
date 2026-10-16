#!/usr/bin/env python3
"""Decode-step tables from INTARSIA_METAL_PROFILE traces (crates/intarsia-backend-metal/src/profile.rs).

usage: analyze.py <profile-dir> <short|4k> <decode-steps>
Reads kernel-<tag>.jsonl (one command buffer per launch) and cb-<tag>.jsonl (stream as normal), takes the last
<decode-steps> decode steps (each starts at an embedding_bf16 record with t == 1) and prints markdown.
"""
import gzip, json, os, sys, collections

d, tag, n_steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
GEMV = ("matmul_bf16_gemv", "logits_bf16_gemv")
ATTN = ("attn_decode_bf16", "kv_append_bf16")

def load(p):
    f = gzip.open(p + ".gz", "rt") if os.path.exists(p + ".gz") else open(p)
    return [json.loads(l) for l in f]

def segments(recs):
    idx = [i for i, r in enumerate(recs) if r["op"] == "embedding_bf16" and r["t"] == 1]
    segs = [recs[a:b] for a, b in zip(idx, idx[1:] + [len(recs)])]
    return segs[-n_steps:]

def cat(op):
    return "GEMV" if op in GEMV else "attention" if op in ATTN else "elementwise/row"

k = load(f"{d}/kernel-{tag}.jsonl")
segs = segments(k)
assert len(segs) == n_steps, (len(segs), n_steps)
tab = collections.OrderedDict()
cats = collections.Counter()
for s in segs:
    for j, r in enumerate(s):
        dur = r["e"] - r["s"]
        key = (r["op"], r["n"], r["k"], r["bytes"])
        t = tab.setdefault(key, [0, 0.0])
        t[0] += 1
        t[1] += dur
        cats[cat(r["op"])] += dur
span = sum(s[-1]["e"] - s[0]["s"] for s in segs)
total = sum(cats.values())
calls_per_step = sum(v[0] for v in tab.values()) / n_steps
print(f"### {tag}: per-kernel trace, {n_steps} decode steps, {calls_per_step:.0f} launches per step\n")
print("| kernel | n x k | calls/token | us/call | ms/token | share | GB/s |\n|---|---|---:|---:|---:|---:|---:|")
for key, (c, t) in sorted(tab.items(), key=lambda kv: -kv[1][1]):
    op, n, kk, by = key
    shape = f"{n}x{kk}" if n else ""
    gbs = f"{by / (t / c) / 1e9:.0f}" if by else ""
    print(f"| {op} | {shape} | {c / n_steps:.0f} | {t / c * 1e6:.1f} | {t / n_steps * 1e3:.3f} | {100 * t / total:.1f}% | {gbs} |")
print(f"\nSum of kernel time {total / n_steps * 1e3:.2f} ms/token; first-start to last-end span (includes the host submit latency between serialized launches) {span / n_steps * 1e3:.2f} ms/token.\n")
print("| class | ms/token | share of kernel time |\n|---|---:|---:|")
for c, t in cats.most_common():
    print(f"| {c} | {t / n_steps * 1e3:.2f} | {100 * t / total:.1f}% |")
g = [(key, v) for key, v in tab.items() if key[0] in GEMV]
gb = sum(key[3] * v[0] for key, v in g) / n_steps
gt = sum(v[1] for _, v in g) / n_steps
print(f"\nGEMV weight bytes {gb / 1e9:.3f} GB/token in {gt * 1e3:.2f} ms/token = {gb / gt / 1e9:.0f} GB/s effective.\n")

c = load(f"{d}/cb-{tag}.jsonl")
steps = c[-n_steps:]
busy = [r["e"] - r["s"] for r in steps]
gp = [steps[i + 1]["s"] - steps[i]["e"] for i in range(len(steps) - 1)]
per = [steps[i + 1]["s"] - steps[i]["s"] for i in range(len(steps) - 1)]
print(f"### {tag}: command-buffer timeline (stream as normal), last {n_steps} records, dispatches per cb {steps[0]['nd']}..{steps[-1]['nd']}\n")
avg = lambda v: sum(v) / len(v)
print(f"- GPU busy per step (cb GPUEnd-GPUStart): {avg(busy) * 1e3:.2f} ms (min {min(busy) * 1e3:.2f}, max {max(busy) * 1e3:.2f})")
print(f"- GPU idle between consecutive step cbs: {avg(gp) * 1e3:.3f} ms avg (min {min(gp) * 1e3:.3f}, max {max(gp) * 1e3:.3f})")
print(f"- step period (start to start): {avg(per) * 1e3:.2f} ms = {1 / avg(per):.1f} tok/s; GPU busy fraction {100 * avg(busy) / avg(per):.1f}%")
