#!/usr/bin/env python3
"""Compare two INTARSIA_DSV41_DUMP_POS dumps (the same prompt run at two chunk sizes): row by row, step by step.

    uv run --no-project --with numpy python tools/dsv41/chunkdiff.py <dir_a> <dir_b> [--pos 100,511] [--what hc_ffn,indexer]

Files are `{step}-o{k}.p{pos}.{dtype}.{row shape}.bin`. For float outputs the relative rms difference ||a-b|| / ||b|| and whether
the bytes are equal; for int outputs (router ids, indexer top-k) the Jaccard of the id sets (-1 padding dropped) and the fraction
of equal entries. Prints, per position, the first layer whose residual stream (`layers.N.hc_*` output 2) differs, the first step
overall that differs, and the lowest indexer / router Jaccard.
"""
import argparse
import os
import re
from collections import defaultdict

import numpy as np


def load(path, dtype):
    raw = np.fromfile(path, dtype=np.uint8)
    if dtype == "f32":
        return raw.view(np.float32).astype(np.float64)
    if dtype == "bf16":
        return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    if dtype == "f16":
        return raw.view(np.float16).astype(np.float64)
    if dtype == "i32":
        return raw.view(np.int32)
    if dtype == "i16":
        return raw.view(np.int16)
    return raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--pos", default="")
    ap.add_argument("--what", default="")
    ap.add_argument("--thresh", type=float, default=1e-3, help="relative rms above which a float output counts as differing")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    want_pos = {int(x) for x in a.pos.split(",") if x}
    want_what = [x for x in a.what.split(",") if x]
    pat = re.compile(r"^(?P<what>.+)-o(?P<k>\d+)\.p(?P<pos>\d+)\.(?P<dt>[a-z0-9]+)\.(?P<shape>[0-9x]*)\.bin$")
    fa, fb = set(os.listdir(a.a)), set(os.listdir(a.b))
    rows = defaultdict(list)  # pos -> [(layer, step-order key, what, k, kind, value, equal)]
    for f in sorted(fa & fb):
        m = pat.match(f)
        if not m:
            continue
        pos = int(m["pos"])
        if (want_pos and pos not in want_pos) or (want_what and not any(w in m["what"] for w in want_what)):
            continue
        x, y = load(os.path.join(a.a, f), m["dt"]), load(os.path.join(a.b, f), m["dt"])
        if x.shape != y.shape:
            continue
        lm = re.search(r"layers\.(\d+)\.", m["what"])
        layer = int(lm[1]) if lm else (-1 if m["what"].startswith("embed") else 999)
        if m["dt"] in ("i32", "i16"):
            sa, sb = set(x[x >= 0].tolist()), set(y[y >= 0].tolist())
            j = len(sa & sb) / max(1, len(sa | sb))
            rows[pos].append((layer, m["what"], int(m["k"]), "ids", j, bool((x == y).all())))
        else:
            den = np.linalg.norm(y) or 1.0
            rows[pos].append((layer, m["what"], int(m["k"]), "float", float(np.linalg.norm(x - y) / den), bool(np.array_equal(x, y))))
    order = {"hc_attn": 0, "attn": 1, "hc_ffn": 2, "ffn": 3}

    def key(r):
        w = r[1].split(".", 2)[-1] if r[0] >= 0 else r[1]
        return (r[0], order.get(w.split(".")[0], 1.5), r[1])

    for pos in sorted(rows):
        rs = sorted(rows[pos], key=key)
        fl = [r for r in rs if r[3] == "float"]
        ids = [r for r in rs if r[3] == "ids"]
        diff = [r for r in fl if not r[5] and r[4] > a.thresh]
        any_bits = [r for r in fl if not r[5]]
        stream = [r for r in fl if re.search(r"\.hc_(attn|ffn)$", r[1]) and r[2] == 2 and not r[5]]
        print(f"=== position {pos}: {len(rs)} outputs compared, {len(any_bits)} float outputs not bit-equal, {len(diff)} above {a.thresh:g}")
        if any_bits:
            r = any_bits[0]
            print(f"  first bit difference: layer {r[0]} {r[1]} o{r[2]} rel {r[4]:.3g}")
        if diff:
            r = diff[0]
            print(f"  first above threshold: layer {r[0]} {r[1]} o{r[2]} rel {r[4]:.3g}")
        if stream:
            r = stream[0]
            print(f"  first residual stream that differs: layer {r[0]} {r[1]} rel {r[4]:.3g}")
            last = [x for x in stream if x[0] == max(s[0] for s in stream)][-1]
            print(f"  last: layer {last[0]} {last[1]} rel {last[4]:.3g}")
        if ids:
            ix = [r for r in ids if "indexer" in r[1]]
            rt = [r for r in ids if ".gate" in r[1]]
            for name, g in (("indexer top-k", ix), ("router ids", rt)):
                if g:
                    mn = min(g, key=lambda r: r[4])
                    meanj = sum(r[4] for r in g) / len(g)
                    print(f"  {name}: {sum(1 for r in g if r[5])}/{len(g)} equal, mean Jaccard {meanj:.4f}, min {mn[4]:.3f} at {mn[1]}")
        if a.verbose:
            for r in rs:
                print(f"    L{r[0]:>3} {r[1]:<44} o{r[2]} {r[3]:<5} {r[4]:.3g} {'=' if r[5] else ''}")


if __name__ == "__main__":
    main()
