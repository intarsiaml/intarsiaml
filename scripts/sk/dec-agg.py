#!/usr/bin/env python3
"""Lane sk3: aggregate dec-window.sh output (raw/<out>/index.txt): per arm, medians over rounds of
1-stream decode tok/s (speed 1k gen 256), ITL p50/p99 (suite speed-plain 1k row), and per concurrency N: aggregate tok/s,
per-stream tok/s (mean ITL = 1000/rate), TTFT.   usage: dec-agg.py <raw/outdir> [baseline-arm]"""
import json, re, statistics as st, sys, os, collections
d = sys.argv[1]; base = sys.argv[2] if len(sys.argv) > 2 else None
R = "/opt/code/deepeval/results"
runs = collections.defaultdict(lambda: collections.defaultdict(list))
for line in open(os.path.join(d, "index.txt")):
    m = re.match(r"sk-(.+)-r(\d+)(\.suite)?: .*results/([^/\s]+?)(/result.json)?(\s|$)", line)
    if m: runs[m.group(1)]["suite" if m.group(3) else "speed"].append(m.group(4))
def med(x): return st.median(x) if x else float("nan")
def band(x): return f"{med(x):.1f} ({min(x):.1f}-{max(x):.1f})" if x else "-"
out = {}
for arm, r in runs.items():
    o = collections.defaultdict(list)
    for rid in r["speed"]:
        j = json.load(open(f"{R}/{rid}/result.json"))["speed"]
        for row in j["rows"]:
            if row["target_prompt_tokens"] == 1024: o["dec1"].append(row["decode_tok_s"]["median"])
        for c in j.get("concurrency", []) or []:
            n = c["streams"]; o[f"agg{n}"].append(c["aggregate_decode_tok_s"])
            o[f"per{n}"].append((c["per_stream_decode_tok_s"] or {}).get("median"))
            o[f"ttft{n}"].append((c["ttft_s"] or {}).get("median"))
    for rid in r["suite"]:
        j = json.load(open(f"{R}/{rid}/result.json"))["speed"]
        for row in j["rows"]:
            if row["target_prompt_tokens"] == 1024:
                o["itl50"].append(row.get("itl_ms_p50")); o["itl99"].append(row.get("itl_ms_p99")); o["suite_dec"].append(row["decode_tok_s"]["median"])
    out[arm] = o
arms = sorted(out)
print("| arm | 1 stream decode tok/s | ITL p50 ms | ITL p99 ms | 4 streams agg (per-stream) | 8 streams agg (per-stream) | TTFT 4 / 8 s |")
print("|---|---|---|---|---|---|---|")
for a in arms:
    o = out[a]
    c = lambda n: f"{band(o[f'agg{n}'])} ({med([x for x in o[f'per{n}'] if x]):.1f})" if o.get(f"agg{n}") else "-"
    tt = " / ".join(f"{med(o[f'ttft{n}']):.2f}" if o.get(f"ttft{n}") else "-" for n in (4, 8))
    print(f"| {a} | {band(o['dec1'])} | {band(o['itl50'])} | {band(o['itl99'])} | {c(4)} | {c(8)} | {tt} |")
if base:
    for a in arms:
        if a == base: continue
        for k in ("dec1", "agg4", "agg8"):
            if out[a].get(k) and out[base].get(k): print(f"{a} vs {base} {k}: {100*(med(out[a][k])/med(out[base][k])-1):+.1f} %")
