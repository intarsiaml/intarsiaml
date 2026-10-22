#!/usr/bin/env python3
"""fx1: compare top-k dumps / goldens offline (KL21(ref || cand) per row, the tool's 21-way definition).

usage: oracle_stats.py REF CAND [CAND ...]
REF / CAND: intarsia-golden-topk-v1 files (vLLM goldens, `--write-self-golden` dumps). Prefill rows only.

Per candidate, over all prompts: mean KL21 on all rows and on scored rows (ChatML system / user turns hidden, the
`turns.rs` rule), the median and the 95 % trimmed mean of the scored rows, the mean on rows where the reference is
confident (top-1 p >= 0.9), top-1 agreement, and the mean signed shift of the reference argmax log-prob
(cand - ref, nats). Run with `uv run --project tools python scripts/fx1-oracle/oracle_stats.py ...` (numpy).
"""
import json
import struct
import sys

import numpy as np

IM_START, IM_END, SYSTEM, USER = 248045, 248046, 8678, 846
K = 20


def load(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
        base = 8 + n
        data = f.read()
    md = h["__metadata__"]
    prompts = json.loads(md["prompts"])

    def t(name):
        e = h[name]
        b, en = e["data_offsets"]
        dt = {"I32": np.int32, "F32": np.float32}[e["dtype"]]
        return np.frombuffer(data[b:en], dtype=dt).reshape(e["shape"])

    out = {}
    for i, p in enumerate(prompts):
        nm = p["name"] if isinstance(p, dict) else str(p)
        out[nm] = (t(f"ids_{i}"), t(f"topk_ids_{i}"), t(f"topk_logprobs_{i}").astype(np.float64))
    return out


def scored_mask(ids):
    n = len(ids)
    s = np.ones(n, bool)
    t = 0
    while t < n:
        if ids[t] != IM_START or t + 1 >= n or ids[t + 1] not in (SYSTEM, USER):
            t += 1
            continue
        rest = np.where(ids[t + 1 :] == IM_END)[0]
        b = t + 1 + rest[0] if len(rest) else n
        s[t:b] = False
        t = b + 1
    return s


def entropy(lp):
    """Entropy of a top-k row with the mass outside the k ids as one bucket."""
    v = lp > -1e29
    p = np.exp(lp[v])
    rest = max(1.0 - p.sum(), 1e-12)
    return float(-(p * lp[v]).sum() - rest * np.log(rest))


def nll(ids, tid, tlp):
    """-log p(actual next token) per row (NaN when the token is outside the row's top-k)."""
    out = np.full(len(ids), np.nan)
    for r in range(len(ids) - 1):
        j = np.where(tid[r] == ids[r + 1])[0]
        if len(j):
            out[r] = -tlp[r, j[0]]
    return out


def temp_slope(rid, rlp, cid, clp, k=int(__import__("os").environ.get("SLOPE_K", "8"))):
    """Per row: least-squares slope of the candidate's log-probs on the reference's over the reference top-k ids that
    the candidate also kept (a sharper candidate reads > 1). NaN when fewer than 4 ids are shared or the row is flat."""
    out = np.full(rid.shape[0], np.nan)
    for r in range(rid.shape[0]):
        cmap = dict(zip(cid[r].tolist(), clp[r].tolist()))
        xs, ys = [], []
        for i in range(k):
            t = int(rid[r, i])
            if t >= 0 and t in cmap:
                xs.append(rlp[r, i])
                ys.append(cmap[t])
        if len(xs) >= 4:
            x = np.array(xs) - np.mean(xs)
            y = np.array(ys) - np.mean(ys)
            if (x * x).sum() > 1e-6:
                out[r] = (x * y).sum() / (x * x).sum()
    return out


def kl21(rid, rlp, cid, clp):
    n = rid.shape[0]
    kl = np.zeros(n)
    shift = np.zeros(n)
    for r in range(n):
        pr = np.exp(rlp[r, :K])
        pr_rest = max(1.0 - pr.sum(), 1e-12)
        valid = cid[r] >= 0
        cmap = dict(zip(cid[r][valid].tolist(), np.exp(clp[r][valid]).tolist()))
        floor = float(np.exp(clp[r][valid][-1]))
        pc = np.array([cmap.get(int(x), floor) for x in rid[r, :K]])
        pc_rest = max(1.0 - pc.sum(), 1e-12)
        kl[r] = (pr * (np.log(pr) - np.log(np.maximum(pc, 1e-30)))).sum() + pr_rest * (np.log(pr_rest) - np.log(pc_rest))
        shift[r] = np.log(max(pc[0], 1e-30)) - rlp[r, 0]
    return kl, shift


def main():
    ref = load(sys.argv[1])
    print(f"ref {sys.argv[1]}")
    for c in sys.argv[2:]:
        cand = load(c)
        KL, SC, CONF, TOP, SH, DH, DN, TS = [], [], [], [], [], [], [], []
        for nm, (ids, rid, rlp) in ref.items():
            if nm not in cand:
                continue
            cids, cid, clp = cand[nm]
            assert (ids == cids).all(), nm
            if (cid[:, 0] < 0).all():
                continue  # decode-only dump
            kl, sh = kl21(rid, rlp, cid, clp)
            m = scored_mask(ids)
            KL.append(kl)
            SC.append(kl[m])
            CONF.append(kl[np.exp(rlp[:, 0]) >= 0.9])
            TOP.append(rid[:, 0] == cid[:, 0])
            SH.append(sh[m])
            DH.append(np.array([entropy(clp[r]) - entropy(rlp[r]) for r in range(len(ids))])[m])
            d = nll(ids, cid, clp) - nll(ids, rid, rlp)
            DN.append(d[m & ~np.isnan(d)])
            ts = temp_slope(rid, rlp, cid, clp)
            TS.append(ts[m & ~np.isnan(ts)])
        kl, sc, conf, top, sh, dh, dn, ts = (np.concatenate(x) for x in (KL, SC, CONF, TOP, SH, DH, DN, TS))
        q = np.sort(sc)[: int(len(sc) * 0.95)]
        print(
            f"{c.split('/')[-1]:<44s} all {kl.mean():.3e} (n={len(kl)})  scored {sc.mean():.3e} (n={len(sc)})  "
            f"median {np.median(sc):.2e}  trim95 {q.mean():.3e}  conf {conf.mean():.3e} (n={len(conf)})  "
            f"top1 {top.mean() * 100:.2f}%  argmax-shift {sh.mean():+.2e}  dH {dh.mean():+.2e} (se {dh.std() / np.sqrt(len(dh)):.1e})  "
            f"dNLL {dn.mean():+.2e} (se {dn.std() / np.sqrt(len(dn)):.1e}, n={len(dn)})  "
            f"slope-1 median {np.median(ts) - 1:+.2e} mean {ts.mean() - 1:+.2e} (se {ts.std() / np.sqrt(len(ts)):.1e}, n={len(ts)})  max {sc.max():.3f}"
        )


if __name__ == "__main__":
    main()
