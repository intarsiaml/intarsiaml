"""W9 summary (lane ink-hip-bpf): the 4-stream filler row, R (packing, 4096 rows) / A (main's batching server), per paired rep and
pooled with W7 / W8's two reps; 95 % t interval. Pairs: W7 (R, A), W8 (A, R) from speed windows; W9 k1 (R1, A1), k2 (A1, R2),
k3 (R2, A2). Decision rule (brief 2b): default-on if the pooled TTFT p50 ratio is <= 1.01 or its interval reaches <= 1.01.
  python3 w9_sum.py"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def row(path, streams=4):
    for c in json.load(open(path))["speed"]["concurrency"]:
        if c["streams"] == streams:
            return c["ttft_s"]["median"], c["aggregate_decode_tok_s"], c["ttft_s"]["max"]
    raise KeyError(path)


def ci(xs):
    n = len(xs)
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
    h = T95[n - 1] * sd / math.sqrt(n)
    return mu, mu - h, mu + h


def main():
    p = lambda *a: os.path.join(HERE, *a)
    pairs = [
        ("W7 (R, A)", p("w7", "R-fill-r1.result.json"), p("w7", "A-fill-r1.result.json")),
        ("W8 (A, R)", p("w8", "R-fill-r1.result.json"), p("w8", "A-fill-r1.result.json")),
        ("W9 k1 (R, A)", p("w9", "R1-k1-r1.result.json"), p("w9", "A1-k1-r1.result.json")),
        ("W9 k2 (A, R)", p("w9", "R2-k2-r1.result.json"), p("w9", "A1-k2-r1.result.json")),
        ("W9 k3 (R, A)", p("w9", "R2-k3-r1.result.json"), p("w9", "A2-k3-r1.result.json")),
    ]
    tt, ag = [], []
    print("| pair | A TTFT p50 s | R TTFT p50 s | R / A TTFT p50 | A agg tok/s | R agg tok/s | R / A agg | TTFT max A / R s |")
    print("|---|---|---|---|---|---|---|---|")
    for name, r, a in pairs:
        if not (os.path.exists(r) and os.path.exists(a)):
            print(f"| {name} | (missing) | | | | | | |")
            continue
        (rt, rg, rm), (at, agg, am) = row(r), row(a)
        tt.append(rt / at)
        ag.append(rg / agg)
        print(f"| {name} | {at:.3f} | {rt:.3f} | {rt / at:.3f} | {agg:.1f} | {rg:.1f} | {rg / agg:.3f} | {am:.2f} / {rm:.2f} |")
    for lab, xs in (("TTFT p50", tt), ("aggregate", ag)):
        if len(xs) >= 2:
            mu, lo, hi = ci(xs)
            print(f"pooled R / A {lab}: {mu:.4f} [{lo:.4f}, {hi:.4f}] (n = {len(xs)})")
    if len(tt) >= 2:
        mu, lo, _ = ci(tt)
        print(f"decision rule: mean <= 1.01: {mu <= 1.01}; interval reaches <= 1.01: {lo <= 1.01} -> "
              f"{'DEFAULT-ON' if (mu <= 1.01 or lo <= 1.01) else 'OPT-IN'}")


if __name__ == "__main__":
    main()
