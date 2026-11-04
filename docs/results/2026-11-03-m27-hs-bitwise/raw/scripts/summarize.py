#!/usr/bin/env python3
"""Summarise the speed chain: per arm decode tok/s (median of 3 reps, min-max) at 4k and 16k, then A / B / C aggregates per mode.
usage: summarize.py > speed-table.md   (reads /opt/code/deepeval/results/*adhoc-m27-<mode>-<tag>/result.json)"""
import glob, json, statistics as st
R = "/opt/code/deepeval/results"
ARMS = {"A": "A: HSB unset (HS=1 baseline)", "B": "B: HSB=3 (bitwise)", "C": "C: HS=3 (reference, not bitwise)"}
def load(mode, tag):
    fs = sorted(glob.glob(f"{R}/*adhoc-m27-{mode}-{tag}/result.json"))
    if not fs: return None
    d = json.load(open(fs[-1]))
    out = {}
    for r in d["speed"]["rows"]:
        out[r["target_prompt_tokens"]] = r
    return fs[-1].split("/")[-2], out
for mode in ("off", "k2"):
    print(f"\n### MTP {mode}\n")
    print("| arm | run | job dir | 4k decode tok/s (med, min-max) | 16k decode tok/s (med, min-max) | 4k prefill | 16k prefill |")
    print("|---|---|---|---|---|---|---|")
    agg = {k: {4096: [], 16384: []} for k in "ABC"}
    for k in "ABC":
        for n in "12":
            tag = f"{k}{n}"; l = load(mode, tag)
            if not l: continue
            name, rows = l
            cells = []
            for sz in (4096, 16384):
                d = rows[sz]["decode_tok_s"]; agg[k][sz].append(d["median"])
                cells.append(f"{d['median']:.2f} ({d['min']:.2f}-{d['max']:.2f})")
            pf = [f"{rows[sz]['prefill_tok_s']['median']:.0f}" for sz in (4096, 16384)]
            print(f"| {k} | {tag} | {name} | {cells[0]} | {cells[1]} | {pf[0]} | {pf[1]} |")
    print("\n| arm | 4k mean of run medians | 16k mean of run medians | 4k vs A | 16k vs A |\n|---|---|---|---|---|")
    base = {sz: st.mean(agg["A"][sz]) for sz in (4096, 16384) if agg["A"][sz]}
    for k in "ABC":
        if not agg[k][4096]: continue
        m = {sz: st.mean(agg[k][sz]) for sz in (4096, 16384)}
        print(f"| {ARMS[k]} | {m[4096]:.2f} | {m[16384]:.2f} | x{m[4096]/base[4096]:.3f} | x{m[16384]/base[16384]:.3f} |")
