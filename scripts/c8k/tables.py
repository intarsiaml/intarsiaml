#!/usr/bin/env python3
"""Lane c8k: tables from the speed window's localeval results. usage: tables.py RAWDIR   (RAWDIR/index.txt lists label: exit N <result.json>)
Prefill tok/s = uncached prompt tokens / TTFT (localeval), median of 3 reps per visit; one row per (arm, size) over the visits, plus the
delta of each intarsia arm against vLLM of the same round and the median over rounds."""
import json, re, statistics, sys, os, collections

raw = sys.argv[1]
res = {}
for line in open(os.path.join(raw, "index.txt")):
    m = re.match(r"(\S+): exit (\d+) (\S+)?", line.strip())
    if m and m.group(3) and os.path.exists(m.group(3)):
        res[m.group(1)] = json.load(open(m.group(3)))
pre = collections.defaultdict(lambda: collections.defaultdict(dict))  # arm -> size -> round -> tok/s
for lab, d in res.items():
    m = re.match(r"c8k-(vllm|i2048|i8192)-r(\d)$", lab)
    if not m:
        continue
    for row in d["speed"]["rows"]:
        if row["prefill_tok_s"]:
            pre[m.group(1)][row["target_prompt_tokens"]][int(m.group(2))] = row["prefill_tok_s"]["median"]
print("| prefill tok/s | " + " | ".join(f"{s}" for s in (1024, 4096, 16384)) + " |\n|---|---:|---:|---:|")
for arm in ("vllm", "i2048", "i8192"):
    cells = []
    for s in (1024, 4096, 16384):
        v = pre[arm][s]
        cells.append(f"{statistics.median(v.values()):.0f} ({', '.join(f'{v[r]:.0f}' for r in sorted(v))})" if v else "-")
    print(f"| {arm} | " + " | ".join(cells) + " |")
print("\n| vs vLLM (same round, median of rounds) | 1024 | 4096 | 16384 |\n|---|---:|---:|---:|")
for arm in ("i2048", "i8192"):
    cells = []
    for s in (1024, 4096, 16384):
        ds = [100 * (pre[arm][s][r] / pre["vllm"][s][r] - 1) for r in pre[arm][s] if r in pre["vllm"][s]]
        cells.append(f"{statistics.median(ds):+.1f} % ({', '.join(f'{x:+.1f}' for x in ds)})" if ds else "-")
    print(f"| {arm} | " + " | ".join(cells) + " |")
print("\n| 4 streams (aggregate decode tok/s, per-stream, TTFT s) | 1k | 4k |\n|---|---|---|")
for arm in ("vllm", "i2048", "i8192"):
    cells = []
    for tag in ("conc1k", "conc4k"):
        a, p, t = [], [], []
        for lab, d in res.items():
            if re.match(rf"c8k-{arm}-{tag}-r\d$", lab) and d["speed"]["concurrency"]:
                c = d["speed"]["concurrency"][0]
                a.append(c["aggregate_decode_tok_s"]); p.append(c["per_stream_decode_tok_s"]["median"]); t.append(c["ttft_s"]["median"])
        cells.append(f"{statistics.median(a):.1f} ({', '.join(f'{x:.1f}' for x in a)}); {statistics.median(p):.1f}; {statistics.median(t):.2f}" if a else "-")
    print(f"| {arm} | " + " | ".join(cells) + " |")
print("\n| ITL, 4 streams of ~4k prompts (ms) | scenario | pooled p50 | pooled p99 | pooled max | agg tok/s |\n|---|---|---:|---:|---:|---:|")
for arm in ("vllm", "i2048", "i8192"):
    for sc in ("burst", "stagger"):
        p50, p99, mx, ag = [], [], [], []
        for r in (1, 2, 3):
            f = os.path.join(raw, f"c8k-{arm}-itl-r{r}.json")
            if os.path.exists(f):
                s = json.load(open(f))["summary"].get(sc)
                if s:
                    p50.append(s["pooled_itl_p50_ms_median"]); p99.append(s["pooled_itl_p99_ms_median"]); mx.append(s["pooled_itl_max_ms_median"]); ag.append(s["aggregate_decode_tok_s_median"])
        if p99:
            f2 = lambda x: f"{statistics.median(x):.0f} ({', '.join(f'{y:.0f}' for y in x)})"
            print(f"| {arm} | {sc} | {f2(p50)} | {f2(p99)} | {f2(mx)} | {statistics.median(ag):.0f} |")
