#!/usr/bin/env python3
"""usage: cmp-acc.py BEFORE.tsv AFTER.tsv [CONTROL.tsv ...] -- per (mode, prompt): rows that differ between the sides, KL21 against the reference on the rows the
peel touches (= differing rows plus all later rows in decode mode), last-16-row KL (prefill), window KL (decode), and totals over the prompts the peel reached."""
import sys, math, collections
def load(p):
    d = collections.OrderedDict()
    for l in open(p):
        f = l.rstrip("\n").split("\t")
        if len(f) < 8: continue
        d.setdefault((f[0], f[1]), []).append((int(f[2]), int(f[3]), int(f[4]), float(f[5]), float(f[6]), float(f[7])))
    return d
B, A = load(sys.argv[1]), load(sys.argv[2]); C = [load(p) for p in sys.argv[3:]]
for c in C: print("control", "identical to BEFORE" if c == B else "DIFFERS from BEFORE (%d keys)" % sum(c[k] != B[k] for k in B))
hdr = f"{'mode':8} {'prompt':7} {'n':>5} {'rows':>5} {'diffrows':>8} {'first':>6} | {'KLb tail16':>11} {'KLa tail16':>11} | {'KLb all':>10} {'KLa all':>10} | {'top1 flips b->a':>15}"
print(hdr)
rows = []
for k in B:
    b, a = B[k], A[k]
    diff = [i for i, (x, y) in enumerate(zip(b, a)) if x != y]
    kl = lambda r, lo: sum(x[3] for x in r[lo:]) / max(1, len(r[lo:]))
    t16 = max(0, len(b) - 16)
    f = diff[0] if diff else None
    rows.append((k, len(b), len(diff), f, kl(b, t16), kl(a, t16), kl(b, 0), kl(a, 0), sum(x[2] for x in b), sum(x[2] for x in a)))
    print(f"{k[0][:8]:8} {k[1]:7} {len(b):5d} {len(b):5d} {len(diff):8d} {str(f):>6} | {kl(b,t16):11.4e} {kl(a,t16):11.4e} | {kl(b,0):10.4e} {kl(a,0):10.4e} | {sum(x[2] for x in b)}->{sum(x[2] for x in a)}")
for mode in ("prefill", "decode"):
    R = [r for r in rows if r[0][0].startswith(mode) and r[2] > 0]
    if not R: print(mode, "no prompt differs"); continue
    n = len(R); db = [r[6] - r[7] for r in R]  # KLb - KLa over the whole mode window (positive = after is closer)
    dt = [r[4] - r[5] for r in R]
    def stat(x):
        m = sum(x) / len(x); sd = (sum((v - m) ** 2 for v in x) / (len(x) - 1)) ** .5 if len(x) > 1 else float("nan"); return m, sd / len(x) ** .5, sum(v > 0 for v in x), sum(v < 0 for v in x)
    for name, x in (("whole window", db), ("last 16 rows", dt)):
        m, se, pos, neg = stat(x)
        print(f"{mode}: {n} prompts reached by the peel; KL(before) - KL(after) over {name}: mean {m:+.3e} +- {se:.1e} (se); after closer on {pos}, before closer on {neg}")
    print(f"{mode}: mean KL21 on those prompts, before {sum(r[6] for r in R)/n:.4e} after {sum(r[7] for r in R)/n:.4e}; last-16 rows before {sum(r[4] for r in R)/n:.4e} after {sum(r[5] for r in R)/n:.4e}")
