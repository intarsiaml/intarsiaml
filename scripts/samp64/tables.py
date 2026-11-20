#!/usr/bin/env python3
"""samp64: markdown tables from the sampled.py JSON files in docs/results/2026-11-20-samp64/raw (python3 scripts/samp64/tables.py [RAW_DIR])."""
import json, pathlib, sys

raw = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "docs/results/2026-11-20-samp64/raw")

def load(win, T):
    f = raw / f"{win}-T{T}-sampled.json"
    if not f.exists():
        return {}
    return {r["size"]: r for r in json.load(open(f))["rows"] if "skipped" not in r}

def cell(r, key="decode_tok_s"):
    return "-" if r is None else f"{r[key]['median']:.1f} ({r[key]['min']:.0f}-{r[key]['max']:.0f})"

def tpv(r):
    m = (r or {}).get("mtp") or {}
    return m.get("tokens_per_verify")

def acc(r):
    m = (r or {}).get("mtp") or {}
    return m.get("accept_rate")

WINS = [("w1", "spec on, k=2 (main)"), ("w2", "`--spec off`"), ("w3", "`--spec-k 1`"), ("w4", "`INTARSIA_SPEC_K_MIN=1`"), ("w5", "`INTARSIA_SPEC_COSTGATE=1`")]
SIZES = [4096, 32768, 65536]
for T, name in [("0.7", "sampled T=0.7"), ("0", "greedy")]:
    print(f"\n### {name}: decode tok/s, median of 5 reps (min-max)\n")
    print("| config | 4k | 32k | 64k |\n|---|---|---|---|")
    for w, lab in WINS:
        d = load(w, T)
        if d:
            print(f"| {lab} | " + " | ".join(cell(d.get(s)) for s in SIZES) + " |")
    print(f"\nMTP acceptance (accepted / proposed) and tokens per verify pass, {name}:\n")
    print("| config | 4k | 32k | 64k |\n|---|---|---|---|")
    for w, lab in WINS:
        d = load(w, T)
        if d and any(acc(d.get(s)) for s in SIZES):
            print(f"| {lab} | " + " | ".join(f"{acc(d.get(s)):.1%}, {tpv(d.get(s)):.2f}" if acc(d.get(s)) else "-" for s in SIZES) + " |")

print("\n### Derived cost per verify pass (ms per pass = tokens per pass x 1000 / tok/s) against the plain step of the spec-off run\n")
print("| | 4k | 32k | 64k |\n|---|---|---|---|")
for T, name in [("0.7", "sampled"), ("0", "greedy")]:
    on, off = load("w1", T), load("w2", T)
    row_p, row_s, row_r = [], [], []
    for s in SIZES:
        if s in on and s in off and tpv(on[s]):
            p = tpv(on[s]) * 1000 / on[s]["decode_tok_s"]["median"]
            pl = 1000 / off[s]["decode_tok_s"]["median"]
            row_p.append(f"{p:.1f}"); row_s.append(f"{pl:.1f}"); row_r.append(f"{p / pl:.2f} (needs {p / pl:.2f}, gets {tpv(on[s]):.2f})")
    print(f"| {name} spec pass, ms | " + " | ".join(row_p) + " |")
    print(f"| {name} plain step, ms | " + " | ".join(row_s) + " |")
    print(f"| {name} pass / plain step (tokens per pass needed to break even, got) | " + " | ".join(row_r) + " |")
