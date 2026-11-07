#!/usr/bin/env python3
"""Lane cgv: rank the kernels of the decode part of a `generate` nsys trace (cuda_gpu_trace csv).

    rank_decode.py trace_cuda_gpu_trace.csv [decode_wall_seconds]
The prefill/decode boundary is the last idle gap >= GAP ms (default 3) before the final N kernels... simpler: the decode window is every kernel
that starts after the last kernel longer than LONG ms (default 4): prefill GEMMs/attention run for tens of ms, a decode kernel never does.
Prints per kernel group ms, share, calls, mean us, plus decode steps (count of the busiest tiny kernel), span and idle share.
"""
import csv, re, sys

def key(name):
    n = name.replace("void ", "")
    n = re.sub(r"^(\(anonymous namespace\)|<unnamed>)::", "", n)
    n = re.sub(r"\(.*", "", n)
    n = re.sub(r"<\(int\)", "<", n).replace("(bool)", "")
    return n[:90]

ev = []
for r in csv.DictReader(open(sys.argv[1])):
    name = r.get("Name") or ""
    if name.startswith("[CUDA"): continue
    s = float(r["Start (ns)"]); ev.append((s, s + float(r["Duration (ns)"]), name))
ev.sort()
if len(sys.argv) > 2:  # decode wall seconds (from the "decode N ms" line of generate): the window is the last that many seconds
    cut = max(e[1] for e in ev) - float(sys.argv[2]) * 1e9
    last = max(i for i, e in enumerate(ev) if e[0] < cut)
else:
    last = max(i for i, e in enumerate(ev) if (e[1] - e[0]) / 1e6 > 4.0)
dec = ev[last + 1:]
print(f"kernels total {len(ev)}, prefill part {last + 1}, decode part {len(dec)}")
t0, t1 = dec[0][0], max(e[1] for e in dec)
groups = {}
for s, e, n in dec:
    g = groups.setdefault(key(n), [0.0, 0]); g[0] += e - s; g[1] += 1
tot = sum(g[0] for g in groups.values())
busy, cs, ce = 0.0, dec[0][0], dec[0][1]
for s, e, _ in dec[1:]:
    if s > ce: busy += ce - cs; cs, ce = s, e
    else: ce = max(ce, e)
busy += ce - cs
print(f"decode span {(t1 - t0) / 1e6:.1f} ms, kernel time {tot / 1e6:.1f} ms, union busy {busy / 1e6:.1f} ms, idle {(t1 - t0 - busy) / 1e6:.1f} ms")
for k, (t, c) in sorted(groups.items(), key=lambda x: -x[1][0])[:30]:
    print(f"{k:<92} {t / 1e6:8.1f} ms {100 * t / tot:5.1f} % {c:7d} calls {t / c / 1e3:9.1f} us avg")
