#!/usr/bin/env python3
"""lane interleave: per-request paired TTFT delta of arms against a reference arm (same conversation and turn = same prompt),
plus each arm's settle split (whole / slices by site) and identical replies.
usage: paired.py RAW_DIR REF_PREFIX ARM_PREFIX [ARM_PREFIX...]   (prefix e.g. F-off-interleaved-seed0-k2)"""
import json, pathlib, statistics, sys, re

raw = pathlib.Path(sys.argv[1])
def load(p):
    return {(r["conv"], r["turn"]): r for r in map(json.loads, open(raw / f"{p}.jsonl"))}
def settles(name):
    log = (raw / f"serve-{name}.log").read_text()
    out = {}
    for m in re.finditer(r"prefix store: settle (slice )?\(([a-z ]+)\) moved (\d+) MiB off the device in ([0-9.]+) ms", log):
        k = ("slice " if m.group(1) else "whole ") + m.group(2)
        n, ms = out.get(k, (0, 0.0))
        out[k] = (n + 1, ms + float(m.group(4)))
    sync = log.count("); host")
    return out, sync
ref = load(sys.argv[2])
print("| arm vs " + sys.argv[2].split("-interleaved")[0] + " | dTTFT sum | dTTFT median / p90 per request | replies identical | settles (n, ms) by site | sync host captures |")
print("|---|---|---|---|---|---|")
for p in sys.argv[3:]:
    a = load(p)
    d = sorted(a[k]["ttft_s"] - ref[k]["ttft_s"] for k in ref)
    same = sum(1 for k in ref if a[k]["reply"] == ref[k]["reply"])
    st, sync = settles(p.split("-interleaved")[0])
    sts = "; ".join(f"{k}: {n}, {ms:.0f}" for k, (n, ms) in sorted(st.items()))
    print(f"| {p.split('-interleaved')[0]} | {sum(d):+.2f} s | {statistics.median(d)*1e3:+.0f} / {d[int(0.9*len(d))]*1e3:+.0f} ms | {same}/{len(ref)} | {sts} | {sync} |")
