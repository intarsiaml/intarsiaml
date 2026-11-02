#!/usr/bin/env python3
"""Lane dsv41/graphs: decode tok/s per text kind from localeval quality runs of the category set.

Usage: catspeed.py LABEL=RESULT_DIR [LABEL=RESULT_DIR ...]

The name prefix of each case is its kind (code, prose, chat, recall). A kind's speed is its total tokens over its total decode
time, so a long reply counts for more than a short one; the median of the per-request speeds is printed beside it.
"""
import json
import statistics as st
import sys
from collections import defaultdict


def load(d):
    r = json.load(open(f"{d}/result.json"))
    out = defaultdict(list)
    for ds in r["datasets"]:
        for c in ds["cases"]:
            s = c.get("speed") or {}
            tps = s.get("decode_tps") or s.get("decode_tok_s")
            n = s.get("completion_tokens")
            if tps and n:
                out[c["name"].split("-")[0]].append((n, tps))
    return out


def main():
    runs = [(a.split("=", 1)[0], load(a.split("=", 1)[1])) for a in sys.argv[1:]]
    kinds = sorted({k for _, r in runs for k in r})
    print(f"{'kind':8s} " + " ".join(f"{name:>24s}" for name, _ in runs))
    for k in kinds + ["all"]:
        cells = []
        for _, r in runs:
            rows = [x for v in r.values() for x in v] if k == "all" else r.get(k, [])
            if not rows:
                cells.append(f"{'-':>24s}")
                continue
            total = sum(n for n, _ in rows) / sum(n / t for n, t in rows)
            cells.append(f"{total:8.1f} (median {st.median(t for _, t in rows):5.1f})".rjust(24))
        print(f"{k:8s} " + " ".join(cells))


if __name__ == "__main__":
    main()
