#!/usr/bin/env python3
"""Verify-step estimate: step(k) = step(1) + dGEMV(k) + 36*(A(k)-A(1)) + dSmall(k); break-even acceptance (iid per drafted token).
usage: estimate.py summary_g0b.json [attn-cand-prefix rows]"""
import sys, json
s = json.load(open(sys.argv[1]))
pref = sys.argv[2] if len(sys.argv) > 2 else "rows"
STEP1 = {128: 27.2, 1024: 27.3, 4096: 28.1}  # ms/token, overnight report (SAFE)
# G0 census-weighted GEMV-only step (36 layers + lm_head), ms: step(R) - step(1)
G0 = {1: 25.53, 2: 25.57, 3: 25.68, 4: 25.70, 5: 25.87, 6: 26.28, 7: 27.43, 8: 30.56}
def t(key, k):
    return s[key.format(k=k)]["t_us"]
def small(k):
    d = lambda n: t("small/0/" + n + "{k}", k) - t("small/0/" + n + "{k}", 1)
    return (36 * (2 * d("norm_add") + d("qkrope") + d("silu")) + d("rmsnorm") + d("argmax") + d("embedding")) / 1000.0
def be(E_target, k):
    lo, hi = 0.0, 1.0
    for _ in range(60):
        a = (lo + hi) / 2
        E = sum(a ** j for j in range(k))
        lo, hi = (a, hi) if E < E_target else (lo, a)
    return (lo + hi) / 2
rows = []
for ctx in (128, 1024, 4096):
    a1 = s[f"attn/{ctx}/serving1"]["t_us"]
    for k in (2, 3, 4, 5, 6, 7, 8):
        ak = s[f"attn/{ctx}/{pref}{k}"]["t_us"]
        dg = G0[k] - G0[1]
        da = 36 * (ak - a1) / 1000.0
        ds = small(k)
        step = STEP1[ctx] + dg + da + ds
        rho = step / STEP1[ctx]
        rows.append((ctx, k, dg, da, ds, step, rho, be(rho, k), rho - 1))
print("| ctx | k | dGEMV (ms) | dAttn (ms) | dSmall (ms) | step(k) (ms) | step(k)/step(1) | break-even alpha | tokens/step needed |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rows:
    print(f"| {r[0]} | {r[1]} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f} | {r[5]:.2f} | {r[6]:.3f} | {r[7]*100:.1f}% | {r[6]:.3f} |")

print()
print("expected tokens per verify step E = (1 - a^k)/(1 - a) and net speed-up E / (step(k)/step(1)), iid per-draft acceptance a")
print("| ctx | k | a=0.3 | a=0.5 | a=0.7 | a=0.9 |")
print("|---|---|---|---|---|---|")
for r in rows:
    if r[1] in (2, 4, 6):
        E = lambda a, k=r[1]: sum(a ** j for j in range(k))
        print(f"| {r[0]} | {r[1]} | " + " | ".join(f"{E(a)/r[6]:.2f}x" for a in (0.3, 0.5, 0.7, 0.9)) + " |")
