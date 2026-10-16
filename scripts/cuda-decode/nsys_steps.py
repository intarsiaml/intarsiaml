#!/usr/bin/env python3
"""Per-decode-step summary of an nsys trace (CUDA graph nodes traced: `nsys profile -t cuda --cuda-graph-trace=node`).

  nsys_steps.py <report.nsys-rep | trace.csv> [--skip N] [--top K] [--json out.json]

Steps are cut at the embedding kernel (every decode step starts with one). For the steady-state steps (the first --skip
steps and any step more than 3x the median span, i.e. the prefill, are dropped) it prints launches per step, the span
from the first kernel start to the last kernel end, the summed kernel time, the idle time inside the span, and the
per-kernel table (calls per step, us per step, average, share of kernel time). The same format as the table of
docs/results/2026-10-14-hip-decode-2.md section 3."""
import argparse, csv, io, json, statistics, subprocess, sys


def load(path):
    if path.endswith(".csv"):
        text = open(path).read()
    else:
        text = subprocess.run(["nsys", "stats", "-r", "cuda_gpu_trace", "-f", "csv", "--force-export=true", path],
                              check=True, capture_output=True, text=True).stdout
    # nsys prints banner lines before the csv header
    lines = text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Start (ns)"))
    rows = list(csv.DictReader(io.StringIO("\n".join(lines[start:]))))
    ks = []
    for r in rows:
        name = r.get("Name", "")
        if not name or name.startswith("[CUDA"):
            continue
        ks.append((int(r["Start (ns)"]), int(r["Duration (ns)"]), name))
    ks.sort()
    return ks


def short(n):
    n = n.replace("(anonymous namespace)::", "").replace("ik_bf16", "bf16").replace("unsigned char", "u8")
    return n if len(n) <= 78 else n[:75] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--skip", type=int, default=5)
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--json")
    a = ap.parse_args()
    ks = load(a.path)
    marks = [i for i, k in enumerate(ks) if "embedding_kernel" in k[2]]
    steps = [ks[m:n] for m, n in zip(marks, marks[1:] + [len(ks)])]
    if not steps:
        sys.exit("no embedding_kernel found: is this a decode trace?")
    spans = [s[-1][0] + s[-1][1] - s[0][0] for s in steps]
    med = statistics.median(spans)
    keep = [s for i, s in enumerate(steps) if i >= a.skip and spans[i] < 3 * med and len(s) > 50]
    if not keep:
        sys.exit("no steady-state steps")
    nk = [len(s) for s in keep]
    span = [(s[-1][0] + s[-1][1] - s[0][0]) / 1e3 for s in keep]
    busy = [sum(k[1] for k in s) / 1e3 for s in keep]
    print(f"steps total {len(steps)}, steady-state {len(keep)} (median span {statistics.median(span):.1f} us)")
    print(f"launches/step {statistics.median(nk):.0f}  span {statistics.median(span):.1f} us  kernel sum {statistics.median(busy):.1f} us  "
          f"idle inside the step {statistics.median(span) - statistics.median(busy):.1f} us "
          f"({100 * (1 - statistics.median(busy) / statistics.median(span)):.1f}%)")
    agg = {}
    for s in keep:
        for _, d, n in s:
            c, t = agg.get(n, (0, 0))
            agg[n] = (c + 1, t + d)
    tot = sum(t for _, t in agg.values()) / len(keep) / 1e3
    rows = sorted(agg.items(), key=lambda kv: -kv[1][1])
    print(f"{'kernel':<80} {'n/step':>7} {'us/step':>9} {'avg us':>8} {'%kern':>6}")
    out = []
    for n, (c, t) in rows[:a.top]:
        per = t / len(keep) / 1e3
        print(f"{short(n):<80} {c / len(keep):7.1f} {per:9.1f} {t / c / 1e3:8.1f} {100 * per / tot:6.1f}")
    rest = sum(t for _, (c, t) in rows[a.top:]) / len(keep) / 1e3
    if rest:
        print(f"(+{len(rows) - a.top} more kernels: {rest:.1f} us/step, {100 * rest / tot:.1f}%)")
    for n, (c, t) in rows:
        out.append({"kernel": n, "per_step": c / len(keep), "us_step": t / len(keep) / 1e3, "avg_us": t / c / 1e3})
    if a.json:
        json.dump({"steps": len(keep), "launches": statistics.median(nk), "span_us": statistics.median(span),
                   "kernel_sum_us": statistics.median(busy), "kernels": out}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
