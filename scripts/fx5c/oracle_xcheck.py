#!/usr/bin/env python3
"""fx5c: cross-check the oracle made on spark1 (aarch64 Linux) against the committed endor oracle (fx1) on the rows both cover:
positions 0..split-1 of long_docs_4k are the same tokens in both (the continue sequence differs after the split).
usage: oracle_xcheck.py COMMITTED_ORACLE FX5C_ORACLE SPLIT"""
import json
import struct
import sys

import numpy as np


def load(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        data = f.read()
    names = [p["name"] if isinstance(p, dict) else str(p) for p in json.loads(h["__metadata__"]["prompts"])]

    def t(name):
        e = h[name]
        b, en = e["data_offsets"]
        return np.frombuffer(data[b:en], dtype={"I32": np.int32, "F32": np.float32}[e["dtype"]]).reshape(e["shape"])

    return names, t


na, ta = load(sys.argv[1])
nb, tb = load(sys.argv[2])
split = int(sys.argv[3])
ia = na.index("long_docs_4k")
ids_a, ids_b = ta(f"ids_{ia}"), tb("ids_0")
assert (ids_a[:split] == ids_b[:split]).all()
ka, la = ta(f"topk_ids_{ia}")[:split], ta(f"topk_logprobs_{ia}")[:split].astype(np.float64)
kb, lb = tb("topk_ids_0")[:split], tb("topk_logprobs_0")[:split].astype(np.float64)
top1 = (ka[np.arange(split), la.argmax(1)] == kb[np.arange(split), lb.argmax(1)]).mean()
kls = []
for j in range(split):
    d = dict(zip(kb[j], lb[j]))
    pa = np.exp(la[j])
    q = np.array([d.get(i, np.nan) for i in ka[j]])
    ok = ~np.isnan(q)
    kl = (pa[ok] * (la[j][ok] - q[ok])).sum()
    kls.append(kl)
kls = np.array(kls)
print(f"rows 0..{split - 1}: top-1 agreement {top1 * 100:.2f}%, KL(endor oracle || spark1 oracle) on shared top-64 ids: mean {kls.mean():.3e} max {kls.max():.3e}; max |dlogprob| on the oracle's top-1: {np.abs(la.max(1) - np.array([dict(zip(kb[j], lb[j])).get(ka[j][la[j].argmax()], np.nan) for j in range(split)])).max():.3e}")
