#!/usr/bin/env python3
"""Markdown tables for the M3 stage 0 doc from cost.json: python3 tools/m3v_tables.py cost.json"""
import json, sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from m3v_project import Costs

path = sys.argv[1]
d = json.load(open(path))
ctxs = ["512", "1024", "4096"]
print("### Component times (ms, median)\n")
print("| pass | " + " | ".join(f"ctx {c}" for c in ctxs) + " |\n|---|" + "---:|" * len(ctxs))
for k in [f"plain{b}" for b in range(1, 7)]:
    print(f"| {k} | " + " | ".join(f"{d['plain'][c][k]['median']:.2f}" for c in ctxs) + " |")
for k in [f"verify{q}" for q in range(2, 7)]:
    print(f"| {k} | " + " | ".join(f"{d['verify'][c][k]['median']:.2f}" for c in ctxs) + " |")
print("\n### Today's iterations: measured composite against the sum of its parts (ms, ctx 512 / 1024 / 4096)\n")
print("| iteration | measured | sum of parts | ratio |\n|---|---|---|---|")
for k in sorted(d["iter"]["512"]):
    cells, sums, rat = [], [], []
    for c in ctxs:
        m = d["iter"][c][k]["median"]
        if k.startswith("iter_vv"):
            n = int(k[7]); s = n * d["verify"][c]["verify5"]["median"]
        else:
            b, kk = int(k[6]), int(k[9]); s = d["verify"][c][f"verify{1+kk}"]["median"] + d["plain"][c][f"plain{b-1}"]["median"]
        cells.append(f"{m:.2f}"); sums.append(f"{s:.2f}"); rat.append(f"{m/s:.3f}")
    print(f"| {k} | " + " / ".join(cells) + " | " + " / ".join(sums) + " | " + " / ".join(rat) + " |")
print("\n### Attention: what a q-row verify group saves against q single-row launches inside a plain step (ms per pass)\n")
print("| q | " + " | ".join(f"g0b-derived ctx {c}" for c in ctxs) + " | " + " | ".join(f"measured P(q)-V(q) ctx {c}" for c in ctxs) + " |\n|---|" + "---:|" * 6)
for q in range(2, 7):
    g = [Costs(path, c, "g0b").attn_corr(q) for c in ctxs]
    m = [Costs(path, c, "meas").attn_corr(q) for c in ctxs]
    print(f"| {q} | " + " | ".join(f"{x:.2f}" for x in g + m) + " |")
print("\n### g0b_bench attn, us per layer (this slot): T1 = serving 1-row; T(q) = rows kernel\n")
print("| ctx | T1 | " + " | ".join(f"T({q})" for q in range(2, 7)) + " |\n|---|---:|" + "---:|" * 5)
for c in ctxs:
    a = d["attn"][c]
    print(f"| {c} | {a['serving1']['us']:.1f} | " + " | ".join(f"{a['rows'+str(q)]['us']:.1f}" for q in range(2, 7)) + " |")
print("\n### One speculating sequence among B (what the brief describes): today vs one batched pass, ms, ctx 512 / 1024 / 4096\n")
print("| B | k | rows R=B+k | today V(1+k)+P(B-1) | batched pass P(R) - corr(g0b) | saving |\n|---|---|---|---|---|---|")
for B in (2, 3):
    for k in (2, 3, 4):
        if B + k > 6:
            print(f"| {B} | {k} | {B+k} | does not fit one pass (R > 6): 2 passes | | |")
            continue
        t, m = [], []
        for c in ctxs:
            cs = Costs(path, c, "g0b")
            t.append(cs.V[1 + k] + cs.P[B - 1]); m.append(cs.pass_cost([1 + k], B - 1))
        print(f"| {B} | {k} | {B+k} | " + " / ".join(f"{x:.1f}" for x in t) + " | " + " / ".join(f"{x:.1f}" for x in m) + " | " + " / ".join(f"{(a-b)/a*100:.0f}%" for a, b in zip(t, m)) + " |")

print("\n### Analytic, not traced and not a pre-registered scenario: one grounded stream beside open-ended stream(s), spec on for the grounded one (k = 4)\n")
print("Per grounded verify pass of t = 3.95 tokens (506 tokens / 128 passes, grounded-n2 counters): the grounded stream verifies in 1 iteration and drains its other t-1 tokens with no row; each open stream needs a plain row in every iteration.\n")
t = 3.95
print("| B | ctx | today ms per cycle | batched-verify ms per cycle | tokens per cycle | today tok/s | batched tok/s | gain |\n|---|---|---|---|---|---|---|---|")
for B in (2, 3):
    for c in ctxs:
        cs = Costs(path, c, "g0b")
        o = B - 1
        today = cs.V[5] + cs.P[o] + (t - 1) * cs.P[o]
        bat = sum(cs.pass_cost(g, pl) for g, pl in __import__("m3v_project").pack([5], o, 6)) + (t - 1) * cs.P[o]
        tok = B * t
        print(f"| {B} | {c} | {today:.1f} | {bat:.1f} | {tok:.2f} | {tok/today*1000:.1f} | {tok/bat*1000:.1f} | {(today/bat-1)*100:+.0f}% |")
