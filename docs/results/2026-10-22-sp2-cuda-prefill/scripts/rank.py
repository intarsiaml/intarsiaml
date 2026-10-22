#!/usr/bin/env python3
"""Lane S1: rank the kernels of one nsys prefill profile and measure the launch gaps of the prompt pass.

    rank.py kern_cuda_gpu_kern_sum.csv kern_cuda_gpu_trace.csv

Kernel names are grouped by their template-stripped name (as scripts/cuda-round6/prof-prefill.sh did). The prompt window
runs from the first kernel to the last kernel of the prompt pass (the profile is `generate --max-tokens 1`: everything);
idle time is span minus the union of kernel intervals. Gaps of 20 us or more are listed separately (cuBLAS lazy init).
"""
import csv
import re
import sys


def key(name: str) -> str:
    n = name.replace("void ", "")
    n = re.sub(r"^(\(anonymous namespace\)|<unnamed>)::", "", n)
    n = re.sub(r"\(.*", "", n)
    n = re.sub(r"<\(int\)", "<", n).replace("(bool)", "")
    return n[:70]


rows = list(csv.DictReader(open(sys.argv[1])))
tot = sum(float(r["Total Time (ns)"]) for r in rows)
groups: dict[str, list[float]] = {}
for r in rows:
    g = groups.setdefault(key(r["Name"]), [0.0, 0])
    g[0] += float(r["Total Time (ns)"])
    g[1] += int(r["Instances"])
print(f"kernel total {tot / 1e6:.1f} ms")
for k, (t, c) in sorted(groups.items(), key=lambda x: -x[1][0])[:24]:
    print(f"{k:<72} {t / 1e6:8.1f} ms {100 * t / tot:5.1f} % {c:6d} calls")

if len(sys.argv) > 2:
    ev = []
    for r in csv.DictReader(open(sys.argv[2])):
        name = r.get("Name") or ""
        if name.startswith("[CUDA"):
            continue
        s = float(r["Start (ns)"])
        ev.append((s, s + float(r["Duration (ns)"]), name))
    ev.sort()
    if ev:
        busy, gaps = 0.0, []
        cur_s, cur_e = ev[0][0], ev[0][1]
        for s, e, n in ev[1:]:
            if s > cur_e:
                busy += cur_e - cur_s
                gaps.append((s - cur_e, n))
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        busy += cur_e - cur_s
        t0, t1 = ev[0][0], max(e for _, e, _ in ev)
        small = [g for g, _ in gaps if g < 20000]
        big = sorted([(g, n) for g, n in gaps if g >= 20000], reverse=True)
        print(f"window: {len(ev)} kernels, span {(t1 - t0) / 1e6:.2f} ms, busy {busy / 1e6:.2f} ms, idle {(t1 - t0 - busy) / 1e6:.2f} ms")
        print(f"gaps < 20 us: {len(small)} totalling {sum(small) / 1e6:.2f} ms (mean {sum(small) / max(1, len(small)) / 1e3:.1f} us); "
              f">= 20 us: {len(big)} totalling {sum(g for g, _ in big) / 1e6:.2f} ms")
        for g, n in big[:5]:
            print(f"  gap {g / 1e3:9.1f} us before {n[:90]}")
