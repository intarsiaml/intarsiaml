#!/usr/bin/env python3
"""Summarise `intarsia bench reuse` runs: store on vs off, per workload (mean of the reps), as a markdown table.
usage: summarize.py RAWDIR   (reads on*.json and off*.json)"""
import glob, json, statistics, sys

d = sys.argv[1]
runs = {k: [json.load(open(f)) for f in sorted(glob.glob(f"{d}/{k}[0-9]*.json"))] for k in ("on", "off")}
print(f"reps: store on {len(runs['on'])}, store off {len(runs['off'])}\n")
print("| workload | requests | prompt tok | computed tok off -> on | TTFT total s off -> on | TTFT mean ms off -> on | TTFT median ms off -> on |")
print("|---|---|---|---|---|---|---|")
for w in ("chat", "agent", "rag"):
    if not runs["on"] or w not in runs["on"][0]["workloads"]:
        continue
    def agg(k, f):
        return statistics.mean(f(r["workloads"][w]) for r in runs[k])
    req = runs["on"][0]["workloads"][w]["requests"]
    pt = runs["on"][0]["workloads"][w]["prompt_tokens"]
    c = [agg(k, lambda x: x["computed_tokens"]) for k in ("off", "on")]
    t = [agg(k, lambda x: x["ttft_ms_total"] / 1e3) for k in ("off", "on")]
    m = [agg(k, lambda x: x["ttft_ms"]["mean"]) for k in ("off", "on")]
    md = [agg(k, lambda x: x["ttft_ms"]["median"]) for k in ("off", "on")]
    print(f"| {w} | {req} | {pt} | {c[0]:.0f} -> {c[1]:.0f} ({100*(c[1]/c[0]-1):+.0f} %) | {t[0]:.2f} -> {t[1]:.2f} ({100*(t[1]/t[0]-1):+.0f} %) | {m[0]:.0f} -> {m[1]:.0f} | {md[0]:.0f} -> {md[1]:.0f} |")
spread = {k: [sum(x["ttft_ms_total"] for x in r["workloads"].values()) / 1e3 for r in runs[k]] for k in ("on", "off")}
print(f"\nall workloads, TTFT total per rep (s): off {spread['off']}, on {spread['on']}")
