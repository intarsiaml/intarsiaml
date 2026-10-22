#!/usr/bin/env python3
"""fx5c: score the dumped engine rows against the CPU oracle.

usage: score.py SEQ.json ORACLE_DUMP.safetensors DUMPDIR [DUMPDIR ...]
DUMPDIR: a `--reuse-dump` directory (cold.f32, warm.f32, warm_pf.f32, cold_engine.f32 when present).

KL(oracle || engine) is the tool's quantity: the oracle's top-64 ids plus one tail bucket (oracle tail = 1 - sum(top-64),
engine tail = engine mass outside the same ids). KL21 (the top-20 + tail partition of fx1's table) is also printed. Full-vocabulary
KL(cold || warm) and KL(warm || cold) are for reference. Row j is the distribution at position row0 + j; the oracle row of the
same position is the reference. Needs numpy.
"""
import json
import struct
import sys

import numpy as np


def load_oracle(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        data = f.read()

    def t(name):
        e = h[name]
        b, en = e["data_offsets"]
        return np.frombuffer(data[b:en], dtype={"I32": np.int32, "F32": np.float32}[e["dtype"]]).reshape(e["shape"])

    return t("ids_0"), t("topk_ids_0"), t("topk_logprobs_0").astype(np.float64)


def logsoftmax(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - (np.log(np.exp(x - m).sum(-1, keepdims=True)) + m)


def kl_support(olp, oid, lq, k):
    """KL(oracle || engine) over the oracle's top-k + tail bucket. olp: [k..] logprobs, lq: engine log-softmax (vocab)."""
    ids, lp = oid[:k], olp[:k]
    p = np.exp(lp)
    ptail = max(1.0 - p.sum(), 0.0)
    lq_top = lq[ids]
    qtail = max(1.0 - np.exp(lq_top).sum(), 0.0)
    kl = (p * (lp - lq_top)).sum()
    if ptail > 1e-12:
        kl += ptail * (np.log(ptail) - np.log(max(qtail, 1e-300)))
    return max(kl, 0.0)


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    seq = json.load(open(sys.argv[1]))
    ids, oid, olp = load_oracle(sys.argv[2])
    assert list(ids) == seq["turn2"] + seq["stream"], "oracle sequence != seq.json"
    row0, m, vocab = seq["row0"], seq["m"], seq["vocab"]
    orows = slice(row0, row0 + m)
    oid, olp = oid[orows], olp[orows]
    o_top1 = oid[np.arange(m), olp.argmax(1)]
    agree = int((o_top1 == np.array(seq["stream"][:m])).sum())
    print(f"oracle rows {row0}..{row0 + m - 1}; oracle top-1 equals the forced stream token (alignment check, row j chose stream[j]) at {agree}/{m} rows")
    out = {}
    for d in sys.argv[3:]:
        eng = {}
        for nm in ("cold", "warm", "warm_pf", "cold_engine"):
            try:
                eng[nm] = np.fromfile(f"{d}/{nm}.f32", dtype="<f4").reshape(m, vocab)
            except FileNotFoundError:
                pass
        lps = {nm: logsoftmax(x) for nm, x in eng.items()}
        res = {}
        for nm, lq in lps.items():
            kl64 = np.array([kl_support(olp[j], oid[j], lq[j], 64) for j in range(m)])
            kl21 = np.array([kl_support(olp[j], oid[j], lq[j], 20) for j in range(m)])
            top1 = np.array([lq[j].argmax() == o_top1[j] for j in range(m)])
            res[nm] = (kl64, kl21, top1)
        print(f"\n== {d}")
        print(f"{'path':<12} {'KL64 mean':>10} {'KL64 max':>10} {'KL21 mean':>10} {'KL21 max':>10} {'top-1 vs oracle':>16}")
        for nm, (kl64, kl21, top1) in res.items():
            print(f"{nm:<12} {kl64.mean():10.3e} {kl64.max():10.3e} {kl21.mean():10.3e} {kl21.max():10.3e} {int(top1.sum()):>10}/{m}")
        if "cold" in res and "warm" in res:
            c, w = res["cold"][0], res["warm"][0]
            print(f"warm/cold oracle-KL ratio: mean {w.mean() / c.mean():.2f}, max {w.max() / c.max():.2f}; rows with warm KL > cold KL: {int((w > c).sum())}/{m}; paired mean diff (warm-cold) {np.mean(w - c):+.2e} +- {np.std(w - c, ddof=1) / np.sqrt(m):.2e} (s.e.)")
            lc, lw = lps["cold"], lps["warm"]
            klcw = np.array([(np.exp(lc[j]) * (lc[j] - lw[j])).sum() for j in range(m)])
            klwc = np.array([(np.exp(lw[j]) * (lw[j] - lc[j])).sum() for j in range(m)])
            flips = int((lc.argmax(1) != lw.argmax(1)).sum())
            print(f"KL(cold||warm) full vocab: mean {klcw.mean():.3e} max {klcw.max():.3e}; KL(warm||cold): mean {klwc.mean():.3e} max {klwc.max():.3e}; argmax flips warm vs cold {flips}/{m}")
            print("row  pos   cold64     warm64     cold21     warm21   oracle-top1 cold warm")
            for j in range(m):
                print(f"{j:>3} {row0 + j:>5} {res['cold'][0][j]:10.3e} {res['warm'][0][j]:10.3e} {res['cold'][1][j]:10.3e} {res['warm'][1][j]:10.3e}   {int(o_top1[j]):>7} {int(res['cold'][2][j])} {int(res['warm'][2][j])}")
        out[d] = {nm: {"kl64": r[0].tolist(), "kl21": r[1].tolist(), "top1": r[2].tolist()} for nm, r in res.items()}
    json.dump(out, open(sys.argv[2] + ".scores.json", "w"))


if __name__ == "__main__":
    main()
