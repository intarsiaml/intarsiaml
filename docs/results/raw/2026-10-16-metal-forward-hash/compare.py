#!/usr/bin/env python3
"""compare.py A.jsonl DUMP_A B.jsonl DUMP_B [label]: row-by-row diff of two fwdrows runs (e.g. chunk 256 vs 2048).
Per prompt: rows whose Model::forward logits hash differs; for those inside the raw tail dump: max abs diff, #logits
differing, argmax flip; plus serving row0 vs forward last row in each run."""
import json, sys, glob, os
from array import array

def rd(p):
    a = array("f")
    with open(p, "rb") as f:
        a.fromfile(f, os.path.getsize(p) // 4)
    return a

def stats(x, y):
    """(max|d|, argmax of d, #differing) of two rows"""
    m, mi, c = 0.0, 0, 0
    for i, (p, q) in enumerate(zip(x, y)):
        if p != q:
            c += 1
            d = abs(p - q)
            if d > m:
                m, mi = d, i
    return m, mi, c

def am(x):
    return max(range(len(x)), key=x.__getitem__)

def load(p):
    return {json.loads(l)["name"]: json.loads(l) for l in open(p)}

def tail(dump, name, L):
    fs = glob.glob(f"{dump}/{name}-fwd-tail*.f32")
    if not fs:
        return None, 0
    a = rd(fs[0])
    n = int(fs[0].rsplit("tail", 1)[1].split(".")[0])
    v = len(a) // n
    return [a[i * v:(i + 1) * v] for i in range(n)], L - n

A, da, B, db = load(sys.argv[1]), sys.argv[2], load(sys.argv[3]), sys.argv[4]
for name in A:
    if name not in B:
        continue
    ra, rb = A[name], B[name]
    L = ra["L"]
    diff = [i for i, (x, y) in enumerate(zip(ra["fwd_rows"], rb["fwd_rows"])) if x != y]
    s = f"{name} L={L} (chunk {ra['chunk']} vs {rb['chunk']}): fwd rows differing {len(diff)}/{L}"
    s += f" {diff if len(diff) <= 20 else diff[:20] + ['...']}"
    s += f"; serve0 equal {ra['serve0'] == rb['serve0']}"
    s += f"; serve0==fwd_last: A {ra['serve0'] == ra['fwd_rows'][-1]} B {rb['serve0'] == rb['fwd_rows'][-1]}"
    print(s)
    ta, f0 = tail(da, name, L)
    tb, _ = tail(db, name, L)
    if ta is None or tb is None:
        continue
    for i in diff:
        k = i - f0
        if 0 <= k < len(ta):
            m, mi, c = stats(ta[k], tb[k])
            print(f"    row {i}: max|d| {m:.3e} at logit {mi} (|v| {abs(ta[k][mi]):.2f}), "
                  f"{c} of {len(ta[k])} logits differ, argmax {am(ta[k])} vs {am(tb[k])}")
    # serving row0 vs forward last row
    for tag, t, d, r in (("A", ta, da, ra), ("B", tb, db, rb)):
        sv = rd(f"{d}/{name}-serve.f32")
        m, mi, c = stats(sv, t[-1])
        print(f"    {tag}: serve row0 vs fwd last row: max|d| {m:.3e}, {c} logits differ")
