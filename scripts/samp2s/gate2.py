#!/usr/bin/env python3
"""samp2s: ABBA gate table from multi.py window JSONs, pooled per arm, with 2 SE of the difference of means (as scripts/samphost/gate.py).
  python3 scripts/samp2s/gate2.py RAW_DIR CELL "none=a1,a4" "fast=a2,a3"      CELL = s2-T0.7-k20-warm, s2-T0.7-k20, s1-T0.7-k0, s2-T0-k-
For each size: combined tok/s over the overlap window (all streams decoding), combined tok/s over the whole request (prefill stagger and tails included),
per-stream tok/s in the overlap, tokens per verify pass (cell counters) and ms per verify pass and sequence = 1000 x tokens per verify / per-stream overlap tok/s."""
import json, math, pathlib, statistics as st, sys

raw = pathlib.Path(sys.argv[1]); cell = sys.argv[2]
arms = [(a.split("=")[0], a.split("=")[1].split(",")) for a in sys.argv[3:]]

def rows(w):
    f = raw / f"{w}-{cell}-multi.json"
    return {r["size"]: r for r in json.load(open(f))["rows"]} if f.exists() else {}

def per_rep(r, key):
    if key == "overlap":
        return [x["combined_overlap"] for x in r["reps"] if "combined_overlap" in x]
    if key == "whole":
        return [x["combined_whole"] for x in r["reps"] if "combined_whole" in x]
    return [v for x in r["reps"] if "per_stream_overlap" in x for v in x["per_stream_overlap"]]

sizes = sorted({s for _, ws in arms for w in ws for s in rows(w)})
for key, title in (("overlap", "combined tok/s over the overlap window"), ("whole", "combined tok/s over the whole request"), ("per", "per-stream tok/s in the overlap")):
    print(f"\n{cell}: {title} (reps)")
    print("| window | " + " | ".join(str(s) for s in sizes) + " |\n|---|" + "---|" * len(sizes))
    pool = {}
    for name, ws in arms:
        for w in ws:
            r = rows(w)
            print(f"| {name} {w} | " + " | ".join(f"{st.mean(per_rep(r[s], key)):.1f} ({len(per_rep(r[s], key))})" if s in r and per_rep(r[s], key) else "-" for s in sizes) + " |")
            for s in sizes:
                if s in r:
                    pool.setdefault((name, s), []).extend(per_rep(r[s], key))
    for name, _ in arms:
        print(f"| **{name} pooled** | " + " | ".join(f"{st.mean(pool[(name, s)]):.1f} ({len(pool[(name, s)])})" if pool.get((name, s)) else "-" for s in sizes) + " |")
    if len(arms) == 2:
        a, b = arms[0][0], arms[1][0]; cells = []
        for s in sizes:
            x, y = pool.get((a, s)), pool.get((b, s))
            if not x or not y or len(x) < 2 or len(y) < 2:
                cells.append("-"); continue
            d = st.mean(y) / st.mean(x) - 1
            se = math.sqrt(st.variance(x) / len(x) + st.variance(y) / len(y)) / st.mean(x)
            cells.append(f"**{d * 100:+.1f} %** (2 SE {2 * se * 100:.1f} %)")
        print(f"| {b} vs {a} | " + " | ".join(cells) + " |")
print("\ntokens per verify pass (cell counters) and ms per verify pass and sequence, per window")
print("| window | " + " | ".join(str(s) for s in sizes) + " |\n|---|" + "---|" * len(sizes))
for name, ws in arms:
    for w in ws:
        r = rows(w); out = []
        for s in sizes:
            if s not in r or not r[s]["tokens_per_verify"] or not per_rep(r[s], "per"):
                out.append("-"); continue
            out.append(f"{r[s]['tokens_per_verify']:.2f} tok, {1000 * r[s]['tokens_per_verify'] / st.mean(per_rep(r[s], 'per')):.1f} ms")
        print(f"| {name} {w} | " + " | ".join(out) + " |")
print("\nsolo top-k / keyed graph / fallback / total spec steps per window (sum over the cell's sizes)")
for name, ws in arms:
    for w in ws:
        r = rows(w); g = lambda k: sum(int(x["counters"].get(k, 0)) for x in r.values())
        print(f"{name} {w}: solo {g('intarsia_spec_verify_topk_solo_total')} graph {g('intarsia_spec_keyed_graph_steps_total')} fallback {g('intarsia_spec_verify_topk_fallback_total')} steps {g('intarsia_spec_steps_total')}")
