#!/usr/bin/env python3
"""Lane G1c: where a decode step's GPU time goes, kernel by kernel, for two nsys traces (e.g. tp=1 graphs vs tp=2
graphs rank 0). Steps are delimited by the argmax kernel; the last N steps (default 48) are averaged.
    kernel_table.py <a cuda_gpu_trace.csv> <b cuda_gpu_trace.csv> [--labels A,B] [--top 25]"""
import collections
import re
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from nsys_split import is_nccl, load  # noqa: E402


def short(n):
    if is_nccl(n):
        return "nccl all-reduce"
    n = n.removeprefix("void ").replace("<unnamed>::", "")
    n = re.sub(r"[<(].*", "", n)  # template args, parameters
    return n.split("::")[-1][:48]


def per_step(rows, steps):
    am = [s for s, d, n in rows if "argmax" in n.lower()][-(steps + 1):]
    lo, hi = am[0], am[-1]
    acc = collections.Counter()
    cnt = collections.Counter()
    for s, d, n in rows:
        if lo <= s < hi:
            k = short(n)
            acc[k] += d / 1e3 / steps
            cnt[k] += 1 / steps
    wall = (hi - lo) / 1e3 / steps
    return acc, cnt, wall


def main():
    a = sys.argv[1:]
    labels, top, steps = ["A", "B"], 25, 48
    if "--labels" in a:
        i = a.index("--labels"); labels = a[i + 1].split(","); del a[i:i + 2]
    if "--top" in a:
        i = a.index("--top"); top = int(a[i + 1]); del a[i:i + 2]
    (pa, ca, wa), (pb, cb, wb) = per_step(load(a[0]), steps), per_step(load(a[1]), steps)
    print(f"step wall (us): {labels[0]} {wa:.0f}, {labels[1]} {wb:.0f}; kernel sum: {sum(pa.values()):.0f}, {sum(pb.values()):.0f}")
    print(f"| kernel | {labels[0]} us/step | {labels[1]} us/step | {labels[1]}/{labels[0]} | launches/step {labels[0]} / {labels[1]} |")
    print("|---|---|---|---|---|")
    keys = sorted(set(pa) | set(pb), key=lambda k: -(pa[k] + pb[k]))
    for k in keys[:top]:
        r = pb[k] / pa[k] if pa[k] else float("inf")
        print(f"| {k} | {pa[k]:.0f} | {pb[k]:.0f} | {r:.2f} | {ca[k]:.0f} / {cb[k]:.0f} |")
    rest_a = sum(pa[k] for k in keys[top:]); rest_b = sum(pb[k] for k in keys[top:])
    print(f"| (other {len(keys) - top}) | {rest_a:.0f} | {rest_b:.0f} | | |")


if __name__ == "__main__":
    main()
