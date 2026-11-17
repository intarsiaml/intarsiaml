#!/usr/bin/env python3
"""Lane sjf: markdown tables from raw/lat.jsonl (ABBA arms fifo-1 shortest-2 shortest-3 fifo-4)."""
import json, statistics, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.startswith("{")]
arm = lambda r: r["arm"].split("-")[0]
def by(scn, f):
    out = {}
    for r in rows:
        if r["scn"] == scn:
            out.setdefault(arm(r), []).append(f(r))
    return out
def med(v): return statistics.median(v)
print("| scenario / metric | fifo | shortest | change |\n|---|---|---|---|")
def line(name, d, fmt="{:.2f}", better_low=True):
    a, b = d.get("fifo", []), d.get("shortest", [])
    if not a or not b: return
    ma, mb = med(a), med(b)
    print(f"| {name} | {fmt.format(ma)} (n={len(a)}: {', '.join(fmt.format(x) for x in a)}) | {fmt.format(mb)} (n={len(b)}: {', '.join(fmt.format(x) for x in b)}) | {100*(mb-ma)/ma:+.1f} % |")
line("pair: short (~1K) TTFT s", by("pair", lambda r: r["short_ttft"]))
line("pair: long (~40K) TTFT s", by("pair", lambda r: r["long_ttft"]))
line("pair: wall s", by("pair", lambda r: r["wall"]))
line("burst: TTFT p50 s (12 requests)", by("burst", lambda r: r["ttft_p50"]))
line("burst: TTFT p99 s", by("burst", lambda r: r["ttft_p99"]))
line("burst: short TTFT p50 s (8)", by("burst", lambda r: r["short_p50"]))
line("burst: short TTFT mean s", by("burst", lambda r: r["short_mean"]))
line("burst: long TTFT p50 s (4)", by("burst", lambda r: r["long_p50"]))
line("burst: long TTFT max s", by("burst", lambda r: r["long_max"]))
line("burst: total s", by("burst", lambda r: r["total"]))
line("decode: aggregate tok/s (4 streams)", by("decode", lambda r: r["agg_tok_s"]))
line("decode: per-stream median tok/s", by("decode", lambda r: statistics.median(r["per_stream_tok_s"])))
