#!/usr/bin/env python3
"""Pairwise distances between INTARSIA_DSV41_DUMP_POS dumps of the same prompt run at different chunk sizes (or other settings).

    uv run --no-project --with numpy python tools/dsv41/chunkstudy.py NAME=DIR [NAME=DIR ...] [--layers 1,5,10,20,30,42]

For every pair of runs: the relative rms difference of the last hidden state and of the logits at the positions both dumped, of the
normalised sublayer input (`layers.L.hc_ffn` output 0) at the chosen layers (mean over the dumped positions), and the mean Jaccard
of the indexer selected sets and of the routed expert ids (mean over layers and positions). Also, per position, the step at
chunk boundaries: values at p-1, p, p+1 around the boundaries 512 / 1024 / 2048 / 4096.
"""
import argparse
import glob
import itertools
import os
import re
from collections import defaultdict

import numpy as np


def load(path):
    m = re.search(r"\.(bf16|f32|f16|i32)\.", os.path.basename(path))
    raw = np.fromfile(path, dtype=np.uint8)
    dt = m[1]
    if dt == "f32":
        return raw.view(np.float32).astype(np.float64)
    if dt == "bf16":
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    if dt == "f16":
        return raw.view(np.float16).astype(np.float64)
    return raw.view(np.int32)


def rel(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) or 1.0))


def jac(a, b):
    sa, sb = set(a[a >= 0].tolist()), set(b[b >= 0].tolist())
    return len(sa & sb) / max(1, len(sa | sb))


def index(d):
    out = {}
    for f in os.listdir(d):
        m = re.match(r"^(?P<what>.+)-o(?P<k>\d+)\.p(?P<pos>\d+)\.", f)
        if m:
            out[(m["what"], int(m["k"]), int(m["pos"]))] = os.path.join(d, f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--layers", default="1,5,10,20,30,42")
    ap.add_argument("--boundary", default="511,512,513,1023,1024,2047,2048,2049,4095,4096")
    a = ap.parse_args()
    runs = {r.split("=", 1)[0]: index(r.split("=", 1)[1]) for r in a.runs}
    names = list(runs)
    layers = [int(x) for x in a.layers.split(",")]
    bpos = [int(x) for x in a.boundary.split(",")]
    allpos = sorted({k[2] for k in runs[names[0]]})
    print("positions dumped:", allpos)
    print("\n### pairwise distances (relative rms unless noted)\n")
    print("| pair | logits @7999 | last hidden @7999 | logits @4095 | n(ffn) mean over positions at layers " + ", ".join(map(str, layers)) + " | indexer Jaccard (mean, min over layer x position) | router-id Jaccard (mean) |")
    print("|---|---|---|---|---|---|---|")
    for x, y in itertools.combinations(names, 2):
        X, Y = runs[x], runs[y]

        def pair(what, k, pos):
            kk = (what, k, pos)
            return (load(X[kk]), load(Y[kk])) if kk in X and kk in Y else None

        cells = []
        for what, pos in (("head", 7999), ("hc_pre-head-", 7999), ("head", 4095)):
            p = pair(what, 0, pos)
            cells.append(f"{rel(*p):.2e}" if p else "-")
        per_layer = []
        for L in layers:
            v = []
            for pos in allpos:
                p = pair(f"layers.{L}.hc_ffn.hc_sublayer", 0, pos)
                if p:
                    v.append(rel(*p))
            per_layer.append(f"{np.mean(v):.2e}" if v else "-")
        ij, rj = [], []
        for (what, k, pos), path in X.items():
            if (what, k, pos) not in Y:
                continue
            if what.endswith(".attn.indexer") and k == 0:
                ij.append(jac(load(path), load(Y[(what, k, pos)])))
            elif what.endswith(".ffn.gate") and k == 1:
                rj.append(jac(load(path), load(Y[(what, k, pos)])))
        print(f"| {x} vs {y} | " + " | ".join(cells) + " | " + " / ".join(per_layer) + f" | {np.mean(ij):.4f}, min {np.min(ij):.3f} | {np.mean(rj):.4f} |" if ij and rj else f"| {x} vs {y} | " + " | ".join(cells) + " | " + " / ".join(per_layer) + " | - | - |")
    print("\n### around the chunk boundaries (mean over layers 1..42 of the relative rms of n(ffn); mean indexer Jaccard) per position\n")
    print("| pair | " + " | ".join(f"p{p}" for p in bpos) + " |")
    print("|---|" + "---|" * len(bpos))
    for x, y in itertools.combinations(names, 2):
        X, Y = runs[x], runs[y]
        row = []
        for pos in bpos:
            v, j = [], []
            for L in range(1, 43):
                kk = (f"layers.{L}.hc_ffn.hc_sublayer", 0, pos)
                if kk in X and kk in Y:
                    v.append(rel(load(X[kk]), load(Y[kk])))
                ki = (f"layers.{L}.attn.indexer", 0, pos)
                if ki in X and ki in Y:
                    j.append(jac(load(X[ki]), load(Y[ki])))
            row.append(f"{np.mean(v):.2e} / {np.mean(j):.3f}" if v and j else (f"{np.mean(v):.2e} / -" if v else "-"))
        print(f"| {x} vs {y} | " + " | ".join(row) + " |")

    # what the chunk boundary itself could break: the window state, Engram, hyper-connection state and compressor of layer 0..1 are exact
    print("\n### bit-equality of the parts with no incoming noise (all pairs, all dumped positions)\n")
    exact = {"layer 0 sparse_attn": "layers.0.attn.sparse_attn", "layer 0 hc_attn n": "layers.0.hc_attn.hc_sublayer", "layer 0 hc_ffn n": "layers.0.hc_ffn.hc_sublayer",
             "layer 0 router weights": "layers.0.ffn.gate", "embedding": "embed"}
    for label, w in exact.items():
        eq = tot = 0
        for x, y in itertools.combinations(names, 2):
            for (what, k, pos), path in runs[x].items():
                if what == w and k == 0 and (what, k, pos) in runs[y]:
                    tot += 1
                    eq += bool(np.array_equal(load(path), load(runs[y][(what, k, pos)])))
        print(f"- {label}: {eq}/{tot} bit-equal")
    eq = tot = 0
    for x, y in itertools.combinations(names, 2):
        for (what, k, pos), path in runs[x].items():
            if what.endswith(".engram.embed") and (what, k, pos) in runs[y]:
                tot += 1
                eq += bool(np.array_equal(load(path), load(runs[y][(what, k, pos)])))
    print(f"- Engram lookups (all Engram layers): {eq}/{tot} bit-equal")
    print("\n### growth with depth (c-pairs, mean over positions of the relative rms of n(ffn); mean router-id Jaccard; mean indexer Jaccard of that layer)\n")
    print("| pair | " + " | ".join(f"L{L}" for L in (0, 1, 2, 3, 5, 8, 12, 16, 20, 30, 40, 42)) + " |")
    print("|---|" + "---|" * 12)
    for x, y in itertools.combinations(names, 2):
        X, Y = runs[x], runs[y]
        row = []
        for L in (0, 1, 2, 3, 5, 8, 12, 16, 20, 30, 40, 42):
            v = [rel(load(X[k]), load(Y[k])) for k in X if k[0] == f"layers.{L}.hc_ffn.hc_sublayer" and k[1] == 0 and k in Y]
            row.append(f"{np.mean(v):.1e}" if v else "-")
        print(f"| {x} vs {y} | " + " | ".join(row) + " |")
    print("\n### each run against the mean of the others (leave-one-out centroid; relative rms of the logits at 7999 and of the last hidden)\n")
    for what, pos in (("head", 7999), ("hc_pre-head-", 7999), ("head", 4095)):
        vecs = {n: load(runs[n][(what, 0, pos)]) for n in names if (what, 0, pos) in runs[n]}
        if len(vecs) < 3:
            continue
        print(f"- {what} @{pos}: " + ", ".join(f"{n} {rel(v, np.mean([u for m, u in vecs.items() if m != n], axis=0)):.3e}" for n, v in vecs.items()))


if __name__ == "__main__":
    main()
