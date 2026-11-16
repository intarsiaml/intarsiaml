#!/usr/bin/env python3
"""itlcreep: re-read the soak2 raw data (docs/results/2026-11-14-soak2/raw) against the server's own per-replay counters.
usage: soak2_recheck.py [RAW_DIR]
Per replay S62..S78 (S61 is the baseline scrape): client TTFT sum (summary.json) vs the server's submit-to-first-token sum (scrape diff),
the server's mean token gap (`intarsia_batch_itl_seconds` sum / count), MTP spec steps, and the client's weighted ITL p99 / p50 medians."""
import json, math, statistics as st, sys

raw = sys.argv[1] if len(sys.argv) > 1 else "docs/results/2026-11-14-soak2/raw"

def parse(f):
    d = {}
    for l in open(f):
        if l.startswith("#"): continue
        p = l.split()
        if len(p) == 2:
            try: d[p[0]] = float(p[1])
            except ValueError: pass
    return d

def slope(xs, y):
    n = len(xs); mx = sum(xs) / n; my = sum(y) / n; sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (v - my) for x, v in zip(xs, y)) / sxx
    se = math.sqrt(sum((v - my - b * (x - mx)) ** 2 for x, v in zip(xs, y)) / (n - 2) / sxx)
    return b, se

def corr(a, b):
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))

M = {s: parse(f"{raw}/serve-S.metrics.after-seed{s}") for s in range(61, 79)}
xs, rows = [], []
print("| seed | client TTFT sum s | server TTFT sum s | client - server s | server mean token gap ms | spec steps | client w99 ms | client w50 ms |")
print("|---|---|---|---|---|---|---|---|")
for s in range(62, 79):
    a, b = M[s - 1], M[s]
    g = lambda k: b[k] - a[k]
    sj = json.load(open(f"{raw}/S{s}-interleaved-seed{s}-k2.summary.json"))
    c, v = sj["totals"]["ttft_s_sum"], g("intarsia_ttft_seconds_sum")
    gap = 1000 * g("intarsia_batch_itl_seconds_sum") / g("intarsia_batch_itl_seconds_count")
    sp = g("intarsia_spec_steps_total"); sm = sj["smoothness"]
    xs.append(s); rows.append((c, v, c - v, gap, sp, sm["itl_ms_p99_median"], sm["itl_ms_p50_median"]))
    print(f"| S{s} | {c:.1f} | {v:.1f} | {c - v:.1f} | {gap:.2f} | {sp:.0f} | {sm['itl_ms_p99_median']:.0f} | {sm['itl_ms_p50_median']:.1f} |")
names = ["client TTFT sum", "server TTFT sum", "client - server", "server mean token gap (ms)", "spec steps", "client w99 (ms)", "client w50 (ms)"]
print("\nStraight-line slope per replay (17 replays S62..S78), value +- standard error:\n")
for i, n in enumerate(names):
    b, se = slope(xs, [r[i] for r in rows]); print(f"- {n}: {b:+.3f} +- {se:.3f}")
col = lambda i: [r[i] for r in rows]
print(f"\nCorrelations: w99 vs spec steps {corr(col(5), col(4)):+.2f}, w50 vs spec steps {corr(col(6), col(4)):+.2f}, w99 vs w50 {corr(col(5), col(6)):+.2f}, client vs server TTFT sum {corr(col(0), col(1)):+.2f}")
