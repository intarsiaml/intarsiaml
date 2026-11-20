#!/usr/bin/env python3
"""memmix: markdown time series of a run, one row per WINDOW minutes (default 15) of the load phase, from the 30 s sampler.
usage: series.py <raw-dir> <tag> [WINDOW]
Columns: mean MemAvailable, engine GPU (nvidia-smi per process), engine RssAnon, store host tier, non-store anon (RssAnon - host tier), their max in the window,
vLLM GPU and cgroup (must not move), requests so far. Stdlib only."""
import csv, pathlib, sys

raw = pathlib.Path(sys.argv[1]); tag = sys.argv[2]; win = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
marks = {}
for line in open(raw / f"marks-{tag}.tsv"):
    t, name = line.strip().split("\t"); marks[name] = int(t)
rows = []
for r in csv.DictReader(open(raw / f"mem-{tag}.csv")):
    try:
        d = {k: float(v) for k, v in r.items()}
    except ValueError:
        continue
    d["host"] = d["prefix_store_tier_bytes_tier__host__"] / 1e6
    d["nonstore"] = d["st_RssAnon"] - d["host"]
    rows.append(d)
t0, t1 = marks["load"], marks.get("idle", rows[-1]["epoch"])
print("| min of load | MemAvailable | engine GPU | RssAnon | store host tier | non-store anon (mean) | non-store anon (max) | vLLM GPU / cgroup | requests (cum.) |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
t = t0
while t < t1:
    e = min(t + win * 60, t1)
    w = [r for r in rows if t <= r["epoch"] < e]
    if w:
        m = lambda k: sum(r[k] for r in w) / len(w)
        print(f"| {(t - t0) / 60:.0f} to {(e - t0) / 60:.0f} | {m('mi_MemAvailable'):.0f} | {m('gpu_mb_intarsia'):.0f} | {m('st_RssAnon'):.0f} | {m('host'):.0f} | {m('nonstore'):.0f} | {max(r['nonstore'] for r in w):.0f} | {m('gpu_mb_vllm'):.0f} / {m('vllm_cg_current'):.0f} | {w[-1]['requests_total']:.0f} |")
    t = e
