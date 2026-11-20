#!/usr/bin/env python3
"""storeprefault gate (c): one row per replay2.py run in RAW_DIR matching a window prefix (stdlib only).
usage: replay_table.py RAW_DIR w5 > table.md
Columns: errors, summed TTFT, TTFT p50 / p90, cached fraction, decode tok/s per stream (median), combined output tok/s, wall, client ITL p99 median / max."""
import glob
import json
import pathlib
import re
import statistics
import sys

raw = pathlib.Path(sys.argv[1])
pref = sys.argv[2]
print("| visit | seed | errors | TTFT sum s | TTFT p50 / p90 s | cached frac | decode tok/s per stream (median) | combined out tok/s | wall s | client ITL p99 ms median / max |")
print("|---|---|---|---|---|---|---|---|---|---|")
for p in sorted(raw.glob(f"{pref}-*-interleaved-seed*-k2.jsonl")):
    m = re.match(rf"({pref}-v\d+-\w+)-interleaved-seed(\d+)-k2", p.name)
    rows = [json.loads(l) for l in open(p)]
    ok = [r for r in rows if r["ok"]]
    summ = json.load(open(p.with_suffix(".summary.json")))
    tt = sorted(r["ttft_s"] for r in ok)
    toks = sum(r["completion_tokens"] for r in ok)
    pt = sum(r["prompt_tokens"] for r in ok)
    ct = sum(r["cached_tokens"] for r in ok)
    p99 = [r["itl_ms_p99"] for r in ok if r.get("itl_ms_p99") is not None]
    print(f"| {m.group(1)} | {m.group(2)} | {len(rows) - len(ok)} | {sum(tt):.1f} | {tt[len(tt) // 2]:.2f} / {tt[int(len(tt) * 0.9)]:.2f} | {ct / pt:.3f} | "
          f"{statistics.median([r['decode_tok_s'] for r in ok]):.1f} | {toks / summ['wall_s']:.1f} | {summ['wall_s']:.1f} | {statistics.median(p99):.0f} / {max(p99):.0f} |")
