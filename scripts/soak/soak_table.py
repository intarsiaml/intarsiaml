#!/usr/bin/env python3
"""soak lane tables. usage: soak_table.py <raw-dir> load|soak
  load: one row per replay2 arm (V-k3, V-k4, L-k3, L-k4): requests, errors, TTFT sum / p50 / p90, decode median per request, wall,
        cached fraction; per-request paired TTFT is not meaningful across engines (different caches), so only aggregates.
  soak: one row per soak replay (S<seed>), in time order, then the sampler drift table (sample-S.tsv) and the pass/fail checks."""
import datetime, glob, json, os, statistics, sys

raw, mode = sys.argv[1], sys.argv[2]


def rows_of(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))] if v else float("nan")


def replay_stats(path):
    r = rows_of(path)
    ok = [x for x in r if x["ok"]]
    tt = [x["ttft_s"] for x in ok if x["ttft_s"] is not None]
    dc = [x["decode_tok_s"] for x in ok if x["decode_tok_s"]]
    pt = sum(x["prompt_tokens"] or 0 for x in ok)
    ct = sum(x["cached_tokens"] or 0 for x in ok)
    t0 = min(datetime.datetime.fromisoformat(x["t_start"]).timestamp() for x in r)
    wall = max(x["end_s"] for x in r)
    return dict(n=len(r), err=len(r) - len(ok), ttft_sum=sum(tt), ttft_p50=statistics.median(tt), ttft_p90=q(tt, 0.9),
                dec_p50=statistics.median(dc), dec_mean=statistics.mean(dc), cached=ct / pt if pt else 0, wall=wall, t0=t0,
                out=sum(x["completion_tokens"] or 0 for x in ok))


if mode == "load":
    print("| arm | requests | errors | TTFT sum s | TTFT p50 s | TTFT p90 s | decode tok/s median | cached frac | wall s | agg output tok/s |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for arm in ("V-k3", "L-k3", "V-k4", "L-k4"):
        for p in sorted(glob.glob(f"{raw}/{arm}-interleaved-seed*-k*.jsonl")):
            s = replay_stats(p)
            print(f"| {arm} | {s['n']} | {s['err']} | {s['ttft_sum']:.1f} | {s['ttft_p50']:.2f} | {s['ttft_p90']:.2f} | {s['dec_p50']:.1f} | {s['cached']:.3f} | {s['wall']:.0f} | {s['out']/s['wall']:.1f} |")
else:
    files = glob.glob(f"{raw}/S*-interleaved-seed*-k2.jsonl")
    st = sorted(((replay_stats(p), os.path.basename(p).split("-")[0]) for p in files), key=lambda x: x[0]["t0"])
    if not st:
        sys.exit("no soak replays")
    base = st[0][0]["t0"]
    print("| replay | start min | requests | errors | TTFT sum s | TTFT p50 s | TTFT p90 s | decode tok/s median (bimodal: see doc) | decode tok/s mean | agg output tok/s | cached frac | wall s |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for s, name in st:
        print(f"| {name} | {(s['t0']-base)/60:.1f} | {s['n']} | {s['err']} | {s['ttft_sum']:.1f} | {s['ttft_p50']:.2f} | {s['ttft_p90']:.2f} | {s['dec_p50']:.1f} | {s['dec_mean']:.1f} | {s['out']/s['wall']:.1f} | {s['cached']:.3f} | {s['wall']:.0f} |")
    for key in ("ttft_sum", "ttft_p50", "dec_p50", "dec_mean", "cached", "wall"):
        v = [s[key] for s, _ in st]
        h = len(v) // 2
        print(f"- {key}: first half mean {statistics.mean(v[:h]):.3f}, second half mean {statistics.mean(v[h:]):.3f}, min {min(v):.3f}, max {max(v):.3f}")
    # sampler
    sp = f"{raw}/sample-S.tsv"
    if os.path.exists(sp):
        L = [l.rstrip("\n").split("\t") for l in open(sp)]
        hdr, data = L[0], [r for r in L[1:] if "NA" not in r[5:8]]  # drop samples taken before the server answered /metrics
        col = {h: i for i, h in enumerate(hdr)}

        def series(name):
            return [float(r[col[name]]) if r[col[name]] not in ("NA", "") else float("nan") for r in data]

        n = len(data)
        print(f"\nsampler: {n} samples, {data[0][0]} to {data[-1][0]}")
        print("| signal | first 5 mean | last 5 mean | min | max | slope/min, all | slope/min, from sample 6 (store full) |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for name, sc in (("mem_avail_mb", 1), ("mem_used_mb", 1), ("swap_used_mb", 1), ("rss_mb", 1), ("intarsia_kv_bytes_used", 1e6),
                         ("intarsia_device_memory_free_bytes", 1e6), ("intarsia_prefix_store_tier_bytes{tier=\"host\"}", 1e6),
                         ("intarsia_prefix_store_entries", 1), ("intarsia_prefix_store_tier_objects{tier=\"host\"}", 1)):
            v = [x / sc for x in series(name)]
            def sl(vv):
                xs = list(range(len(vv)))
                mx, my = statistics.mean(xs), statistics.mean(vv)
                return sum((a - mx) * (b - my) for a, b in zip(xs, vv)) / max(1e-9, sum((a - mx) ** 2 for a in xs))
            print(f"| {name} | {statistics.mean(v[:5]):.1f} | {statistics.mean(v[-5:]):.1f} | {min(v):.1f} | {max(v):.1f} | {sl(v):.2f} | {sl(v[5:]):.2f} |")
        last = data[-1]
        g = lambda k: last[col[k]]
        K_HEV = 'intarsia_prefix_store_tier_evictions_total{tier="host"}'
        K_FL = 'intarsia_requests_finished_total{reason="length"}'
        K_FS = 'intarsia_requests_finished_total{reason="stop"}'
        budget = float(g('intarsia_prefix_store_tier_budget_bytes{tier="host"}'))
        peak = max(series('intarsia_prefix_store_tier_bytes{tier="host"}'))
        print(f"\n- store host bytes peak {peak/1e6:.0f} MB against budget {budget/1e6:.0f} MB ({'within' if peak <= budget else 'OVER'} budget)")
        print(f"- lookups {g('intarsia_prefix_store_lookups_total')}, hits {g('intarsia_prefix_store_hits_total')}, misses {g('intarsia_prefix_store_misses_total')}, "
              f"host evictions {g(K_HEV)}, rejected {g('intarsia_prefix_store_rejected_total')}, inserts {g('intarsia_prefix_store_inserts_total')}")
        print(f"- requests {g('intarsia_requests_total')}, rejected {g('intarsia_requests_rejected_total')}, engine_failed {g('intarsia_engine_failed')}, "
              f"no_progress_s {g('intarsia_engine_no_progress_seconds')}, finished length/stop {g(K_FL)}/{g(K_FS)}")
        # first eviction time
        ev = series('intarsia_prefix_store_tier_evictions_total{tier="host"}')
        first = next((i for i, x in enumerate(ev) if x and x > 0), None)
        print(f"- first host eviction seen in sample {first} (of {n}; 60 s apart)" if first is not None else "- no host eviction seen")
        gaps = [float(r[col['intarsia_batch_itl_seconds_count']]) for r in data]
        b25 = [float(r[col['intarsia_batch_itl_seconds_bucket{le="0.25"}']]) for r in data]
        b50 = [float(r[col['intarsia_batch_itl_seconds_bucket{le="0.5"}']]) for r in data]
        print(f"- server gaps (batch_itl) total {gaps[-1]:.0f}: above 0.25 s {gaps[-1]-b25[-1]:.0f}, above 0.5 s {gaps[-1]-b50[-1]:.0f}")
