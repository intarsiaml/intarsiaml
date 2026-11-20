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

# ---- gate run (fixed binary, windows w8..w15) -------------------------------------------------------------------------------------------------
import statistics

def reps(win, T, size):
    f = raw / f"{win}-T{T}-sampled.json"
    if not f.exists():
        return []
    for r in json.load(open(f))["rows"]:
        if r.get("size") == size and "decode_each" in r:
            return r["decode_each"]
    return []

def line(label, wins, T, sizes):
    cells = []
    for s in sizes:
        v = [x for w in wins for x in reps(w, T, s)]
        cells.append(f"{statistics.mean(v):.1f} (med {statistics.median(v):.1f}, n {len(v)})" if v else "-")
    return f"| {label} | " + " | ".join(cells) + " |"

print("\n### Gate run: sampled T=0.7 decode tok/s, mean of all reps (window median, n reps)\n")
G = [4096, 32768, 65536]
print("| window | 4k | 32k | 64k |\n|---|---|---|---|")
for lab, ws in [("none, w8", ["w8"]), ("none again, w11", ["w11"]), ("none, w8 + w11 pooled", ["w8", "w11"]), ("`16384:1`, w9", ["w9"]), ("`16384:1,40000:0`, w10 (32k, 64k only)", ["w10"])]:
    print(line(lab, ws, "0.7", G))
print("\n### Gate run: greedy decode tok/s (not limited by the schedule)\n")
print("| window | 4k | 32k | 64k |\n|---|---|---|---|")
for lab, ws in [("none, w8", ["w8"]), ("`16384:1`, w9", ["w9"])]:
    print(line(lab, ws, "0", G))
print("\n### Mid-context sampled cells (follow-up job)\n")
MS = [8192, 16384, 24576, 32768, 49152]
print("| window | 8k | 16k | 24k | 32k | 49k |\n|---|---|---|---|---|---|")
for lab, ws in [("none, w12", ["w12"]), ("k=1 everywhere (`1:1`), w15", ["w15"])]:
    print(line(lab, ws, "0.7", MS))
print(line("`16384:1,40000:0`, w10 (32k, 49k only)", ["w10"], "0.7", MS))

# ---- final gate (u1..u4, schedule 28000:1,56000:0 against none, ABBA) -------------------------------------------------------------------------
print("\n### Final gate: schedule `28000:1,56000:0` against none, ABBA (u1 none, u2 S, u3 S, u4 none)\n")
print("sampled T=0.7 decode tok/s: mean of all reps (window median, n reps)\n")
print("| window | 4k | 32k | 64k |\n|---|---|---|---|")
for lab, ws in [("none u1", ["u1"]), ("S u2", ["u2"]), ("S u3", ["u3"]), ("none u4", ["u4"]), ("none pooled (u1 + u4)", ["u1", "u4"]), ("S pooled (u2 + u3)", ["u2", "u3"])]:
    print(line(lab, ws, "0.7", G))
base = {s: [x for w in ("u1", "u4") for x in reps(w, "0.7", s)] for s in G}
cand = {s: [x for w in ("u2", "u3") for x in reps(w, "0.7", s)] for s in G}
if all(base[s] and cand[s] for s in G):
    def se(v):
        return statistics.pstdev(v) / len(v) ** 0.5
    print("\n| sampled | change of the mean (S over none) | 2 SE of the difference |\n|---|---|---|")
    for s, lab in zip(G, ["4k", "32k", "64k"]):
        d = statistics.mean(cand[s]) / statistics.mean(base[s]) - 1
        e = 2 * (se(cand[s]) ** 2 + se(base[s]) ** 2) ** 0.5 / statistics.mean(base[s])
        print(f"| {lab} | {d:+.1%} | {e:.1%} |")
print("\ngreedy decode tok/s (not limited by the schedule): mean of reps (window median, n reps)\n")
print("| window | 4k | 32k | 64k |\n|---|---|---|---|")
for lab, ws in [("none u1", ["u1"]), ("S u2", ["u2"])]:
    print(line(lab, ws, "0", G))
