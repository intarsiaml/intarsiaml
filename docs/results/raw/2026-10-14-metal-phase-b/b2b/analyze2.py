#!/usr/bin/env python3
"""Decode-step tables from INTARSIA_METAL_PROFILE traces, for streams that commit several command buffers per step.

usage: analyze2.py <profile-dir> <tag> <short|4k> <decode-steps>
Reads <tag>-kernel-<ctx>.jsonl[.gz] (one command buffer per launch) and <tag>-cb-<ctx>.jsonl (stream as normal) from
<profile-dir>. The kernel trace gives launches per step and per-kernel GPU time; the cb trace is cut into the last
<decode-steps> steps by dispatch count (the cb records carry `nd`, dispatches per command buffer) and gives the period
(first start to last end), the GPU busy time and the idle time between command buffers.
"""
import collections, gzip, json, os, sys

d, tag, ctx, n_steps = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
GEMV = ("matmul_bf16_gemv", "matmul_pair_bf16_gemv", "logits_bf16_gemv")
ATTN = ("attn_decode_bf16", "kv_append_bf16")


def load(p):
    f = gzip.open(p + ".gz", "rt") if os.path.exists(p + ".gz") else open(p)
    return [json.loads(l) for l in f]


def cat(op):
    return "GEMV" if op in GEMV else "attention (decode + kv_append)" if op in ATTN else "elementwise/row"


k = load(f"{d}/{tag}-kernel-{ctx}.jsonl")
idx = [i for i, r in enumerate(k) if r["op"] == "embedding_bf16" and r["t"] == 1]
segs = [k[a:b] for a, b in zip(idx, idx[1:] + [len(k)])][-n_steps:]
assert len(segs) == n_steps, (len(segs), n_steps)
tab = collections.OrderedDict()
cats = collections.Counter()
ncat = collections.Counter()
for s in segs:
    for r in s:
        dur = r["e"] - r["s"]
        t = tab.setdefault((r["op"], r["n"], r["k"], r["bytes"]), [0, 0.0])
        t[0] += 1
        t[1] += dur
        cats[cat(r["op"])] += dur
        ncat[cat(r["op"])] += 1
launches = sum(v[0] for v in tab.values()) / n_steps
total = sum(cats.values())
print(f"### {tag} @ {ctx}: per-kernel trace, {n_steps} decode steps, {launches:.0f} launches per step\n")
print("| kernel | n x k | calls/token | us/call | ms/token | share | GB/s |\n|---|---|---:|---:|---:|---:|---:|")
for (op, n, kk, by), (c, t) in sorted(tab.items(), key=lambda kv: -kv[1][1]):
    shape = f"{n}x{kk}" if n else ""
    gbs = f"{by / (t / c) / 1e9:.0f}" if by else ""
    print(f"| {op} | {shape} | {c / n_steps:.0f} | {t / c * 1e6:.1f} | {t / n_steps * 1e3:.3f} | {100 * t / total:.1f}% | {gbs} |")
print(f"\nSum of kernel time {total / n_steps * 1e3:.2f} ms/token (isolated command buffers).\n")
print("| class | launches/token | ms/token | share of kernel time |\n|---|---:|---:|---:|")
for c, t in cats.most_common():
    print(f"| {c} | {ncat[c] / n_steps:.0f} | {t / n_steps * 1e3:.2f} | {100 * t / total:.1f}% |")

c = load(f"{d}/{tag}-cb-{ctx}.jsonl")
want = int(round(launches)) * n_steps
acc, j = 0, len(c)
while acc < want and j > 0:
    j -= 1
    acc += c[j]["nd"]
steps = c[j:]
span = steps[-1]["e"] - steps[0]["s"]
busy = sum(r["e"] - r["s"] for r in steps)
gaps = [steps[i + 1]["s"] - steps[i]["e"] for i in range(len(steps) - 1)]
print(f"\n### {tag} @ {ctx}: command-buffer timeline (stream as normal): last {len(steps)} command buffers = {n_steps} steps ({acc} dispatches)\n")
print(f"- step period (first start to last end / steps): {span / n_steps * 1e3:.2f} ms = {n_steps / span:.1f} tok/s")
print(f"- GPU busy {busy / n_steps * 1e3:.2f} ms/step ({100 * busy / span:.1f}%), idle {(span - busy) / n_steps * 1e3:.3f} ms/step")
if gaps:
    print(f"- gaps between consecutive command buffers: mean {sum(gaps) / len(gaps) * 1e6:.1f} us, max {max(gaps) * 1e6:.0f} us, n={len(gaps)}")
