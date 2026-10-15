#!/usr/bin/env python3
"""usage: long_cmp.py <bf16.err> <fp8.err> <ctx>: top-50 logit comparison at the last prompt position (KL of the fp8 run's
renormalised top-50 distribution from the bf16 run's, over the union-free common support) + greedy-id agreement."""
import re, sys, math
def parse(path):
    logits = None; ids = None
    for l in open(path):
        if "top" in l and ":" in l and logits is None:
            toks = re.findall(r"(\d+):(-?\d+\.\d+)", l)
            if len(toks) >= 10: logits = {int(i): float(v) for i, v in toks}
        if l.startswith("ids:"): ids = [int(x) for x in re.findall(r"-?\d+", l)]
    return logits, ids
a, b = parse(sys.argv[1]), parse(sys.argv[2]); ctx = sys.argv[3]
(la, ia), (lb, ib) = a, b
if not la or not lb:
    print(f"ctx {ctx}: no logits line parsed"); sys.exit(0)
common = [i for i in la if i in lb]
def lsm(d, keys):
    m = max(d[k] for k in keys); z = math.log(sum(math.exp(d[k] - m) for k in keys)) + m
    return {k: d[k] - z for k in keys}
pa, pb = lsm(la, common), lsm(lb, common)
kl = sum(math.exp(pa[k]) * (pa[k] - pb[k]) for k in common)
top_a, top_b = max(la, key=la.get), max(lb, key=lb.get)
maxd = max(abs(la[k] - lb[k]) for k in common)
div = next((i for i in range(min(len(ia), len(ib))) if ia[i] != ib[i]), None) if ia and ib else None
print(f"ctx {ctx}: top-1 {'same' if top_a == top_b else 'DIFFERENT'} ({top_a}/{top_b}); {len(common)}/50 common ids; KL(bf16||fp8) over the common top-k {kl:.3e}; max |dlogit| {maxd:.3f}; greedy ids (129): " + (f"identical" if div is None and ia == ib else f"first divergence at token {div}"))
