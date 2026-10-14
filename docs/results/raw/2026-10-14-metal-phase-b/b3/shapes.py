#!/usr/bin/env python3
"""shapes.py <profile-dir> <label>...: per-shape GEMV us/call and GB/s from kernel-mode traces (last 100 decode steps),
split into the 8-mod-16 shard-1 layers (a8 kernels; layers 0-6) and the 16-byte-aligned layers (8-35).
Labels that differ only by a trailing digit are averaged (repeat runs)."""
import collections, gzip, json, statistics, sys

d, labels = sys.argv[1], sys.argv[2:]
PER_LAYER = {("matmul_bf16_gemv", 4096, 4096): 2, ("matmul_pair_bf16_gemv", 2048, 4096): 1,
             ("matmul_pair_bf16_gemv", 24576, 4096): 1, ("matmul_bf16_gemv", 4096, 12288): 1}
NAME = {("matmul_bf16_gemv", 4096, 4096): "q,o 4096x4096", ("matmul_pair_bf16_gemv", 2048, 4096): "k+v pair 2x1024x4096",
        ("matmul_pair_bf16_gemv", 24576, 4096): "gate+up pair 2x12288x4096", ("matmul_bf16_gemv", 4096, 12288): "down 4096x12288",
        ("logits_bf16_gemv", 151936, 4096): "lm_head 151936x4096"}


def load(label):
    k = [json.loads(l) for l in gzip.open(f"{d}/{label}-kernel-short.jsonl.gz", "rt")]
    idx = [i for i, r in enumerate(k) if r["op"] == "embedding_bf16" and r["t"] == 1]
    segs = [k[a:b] for a, b in zip(idx, idx[1:] + [len(k)])][-100:]
    out = collections.defaultdict(lambda: collections.defaultdict(list))
    total = 0.0
    for s in segs:
        cnt = collections.Counter()
        for r in s:
            key = (r["op"], r["n"], r["k"])
            dur = r["e"] - r["s"]
            total += dur
            if key in NAME:
                out[key][cnt[key]].append((dur, r["bytes"]))
                cnt[key] += 1
    return out, total / len(segs)


res = {}
for l in labels:
    res[l] = load(l)
print("| shape | layers | calls/token | " + " | ".join(f"{l} us/call (GB/s)" for l in labels) + " |")
print("|---|---|---:|" + "---:|" * len(labels))
for key, name in NAME.items():
    pl = PER_LAYER.get(key, 0)
    groups = [("all", None)] if pl == 0 else [("0-6 (a8)", range(0, 7 * pl)), ("8-35 (16-byte)", range(8 * pl, 36 * pl)), ("all", range(0, 36 * pl))]
    for gname, rng in groups:
        cells = []
        for l in labels:
            out = res[l][0][key]
            ords = list(out) if rng is None else [o for o in out if o in rng]
            durs = [x[0] for o in ords for x in out[o]]
            by = out[ords[0]][0][1]
            m = statistics.median(durs)
            cells.append(f"{m * 1e6:.1f} ({by / m / 1e9:.0f})")
        n = len(ords)
        print(f"| {name} | {gname} | {n} | " + " | ".join(cells) + " |")
print("\nkernel-time sum per token (ms, isolated command buffers): " + ", ".join(f"{l} {res[l][1] * 1e3:.2f}" for l in labels))
