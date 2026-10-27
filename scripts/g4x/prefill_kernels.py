#!/usr/bin/env python3
"""Per-kernel time of the PREFILL part of an nsys gpu-trace csv of `generate` (everything before the last 65 steps, i.e. before the
64 decode steps + the first decoded token): prefill_kernels.py trace_cuda_gpu_trace.csv [--top K]"""
import csv, io, sys, collections
path = sys.argv[1]
top = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 25
text = open(path).read().splitlines()
start = next(i for i, l in enumerate(text) if l.startswith("Start (ns)"))
rows = list(csv.DictReader(io.StringIO("\n".join(text[start:]))))
ks = sorted((int(r["Start (ns)"]), int(r["Duration (ns)"]), r["Name"]) for r in rows if r.get("Name") and not r["Name"].startswith("[CUDA"))
marks = [i for i, k in enumerate(ks) if "embedding_scaled_kernel" in k[2] or "embedding_kernel" in k[2]]
cut = marks[-65] if len(marks) > 65 else len(ks)
pre = ks[:cut]
tot = sum(k[1] for k in pre)
span = (pre[-1][0] + pre[-1][1] - pre[0][0]) if pre else 0
print(f"prefill: {len(pre)} launches, kernel sum {tot/1e6:.1f} ms, span {span/1e6:.1f} ms, {len(marks)-65} embedding launches (chunks)")
agg = collections.defaultdict(lambda: [0, 0])
for _, d, n in pre:
    a = agg[n[:110]]; a[0] += 1; a[1] += d
for n, (c, d) in sorted(agg.items(), key=lambda x: -x[1][1])[:top]:
    print(f"{c:7d} {d/1e6:9.1f} ms {100*d/tot:5.1f}%  {n}")
