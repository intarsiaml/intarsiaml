#!/usr/bin/env python3
"""soak2 comparison table: pooled soak statistics of several runs next to each other (different windows, boxes and settings: NOT paired).
usage: compare.py <repo-root>   (the three runs are hard-wired below; metrics = last per-replay scrape)"""
import glob, json, re, statistics, sys

root = sys.argv[1]
RUNS = [
    ("soak (4 GiB stress store, no MTP, greedy), spark2, vLLM down, 2026-11-13", "docs/results/2026-11-13-cutover-soak/raw", "S", "serve-S.metrics"),
    ("restore-acc soak (8 GiB default, no MTP, greedy), spark2, vLLM down, 2026-11-13", "docs/results/2026-11-13-restore-acc/raw", "S", "serve-S.metrics.after-seed45"),
    ("soak2 (8 GiB default, MTP on, half T=1.0), spark1, vLLM up, 2026-11-14", "docs/results/2026-11-14-soak2/raw", "S", "serve-S.metrics"),
]


def rows_of(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def metrics(p):
    d = {}
    for l in open(p):
        if l.startswith("#") or not l.strip():
            continue
        k, _, v = l.rpartition(" ")
        try:
            d[k] = float(v)
        except ValueError:
            pass
    return d


print("| run | replays | requests | errors | TTFT sum s per replay (mean, min-max) | TTFT p50 / p90 / p99 s | decode tok/s mean / median | client ITL p99 ms median / max | agg output tok/s (mean of replays) | cached frac | wall s per replay | engine gaps >0.25 s / >0.5 s of total |")
print("|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---|")
for label, d, pre, mf in RUNS:
    fs = [p for p in glob.glob(f"{root}/{d}/{pre}[0-9]*-interleaved-seed*-k2.jsonl")]
    if not fs:
        print(f"| {label} | (no files) |"); continue
    allr, sums, aggs, walls = [], [], [], []
    for p in fs:
        r = rows_of(p)
        ok = [x for x in r if x["ok"]]
        allr += r
        sums.append(sum(x["ttft_s"] for x in ok if x["ttft_s"] is not None))
        wall = max(x["end_s"] for x in r)
        walls.append(wall)
        aggs.append(sum(x["completion_tokens"] or 0 for x in ok) / wall)
    ok = [x for x in allr if x["ok"]]
    tt = [x["ttft_s"] for x in ok if x["ttft_s"] is not None]
    dc = [x["decode_tok_s"] for x in ok if x["decode_tok_s"]]
    p99 = [x["itl_ms_p99"] for x in ok if x.get("itl_ms_p99") is not None]
    pt, ct = sum(x["prompt_tokens"] or 0 for x in ok), sum(x["cached_tokens"] or 0 for x in ok)
    m = metrics(f"{root}/{d}/{mf}")
    n = m.get("intarsia_batch_itl_seconds_count", 0)
    g25 = n - m.get('intarsia_batch_itl_seconds_bucket{le="0.25"}', n)
    g50 = n - m.get('intarsia_batch_itl_seconds_bucket{le="0.5"}', n)
    print(f"| {label} | {len(fs)} | {len(allr)} | {len(allr)-len(ok)} | {statistics.mean(sums):.1f} ({min(sums):.1f}-{max(sums):.1f}) | {statistics.median(tt):.2f} / {q(tt,.9):.2f} / {q(tt,.99):.2f} | "
          f"{statistics.mean(dc):.1f} / {statistics.median(dc):.1f} | {statistics.median(p99):.0f} / {max(p99):.0f} | {statistics.mean(aggs):.1f} | {ct/pt:.3f} | {statistics.mean(walls):.0f} | {g25:.0f} / {g50:.0f} of {n:.0f} |")
