#!/usr/bin/env python3
"""Full-vocabulary distance of the final logits between INTARSIA_DSV41_DUMP_POS dumps of the same prompt (see chunkstudy.py).

    uv run --no-project --with numpy python tools/dsv41/logitdist.py NAME=DIR0:DIR1 [NAME=DIR0:DIR1 ...]

DIR0 / DIR1 are the rank 0 / rank 1 dump directories of one run; the head step writes each rank's half of the vocabulary
(`head-o0.p<pos>.f32.<n>.bin`, one row for the last position of a chunk), so the two halves are concatenated. The first run is the
reference. For every other run and every position both have: relative rms of the logit row, KL(ref || run) in nats, mean |logprob
difference| over the reference's 5 most likely tokens, whether the argmax is the same. Prints per position and the mean.
"""
import glob
import os
import re
import sys

import numpy as np


def load(spec):
    d0, d1 = spec.split(":")
    out = {}
    for f in glob.glob(os.path.join(d0, "head-o0.p*.f32.*.bin")):
        m = re.search(r"\.p(\d+)\.f32\.", f)
        g = os.path.join(d1, os.path.basename(f))
        if m and os.path.exists(g):
            out[int(m[1])] = np.concatenate([np.fromfile(f, np.float32), np.fromfile(g, np.float32)]).astype(np.float64)
    return out


def logsm(x):
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def main():
    runs = [(a.split("=")[0], load(a.split("=", 1)[1])) for a in sys.argv[1:]]
    ref_name, ref = runs[0]
    print(f"reference {ref_name}: positions {sorted(ref)}, vocabulary {len(next(iter(ref.values())))}\n")
    print("| run | position | relative rms | KL(ref||run) nats | mean abs dlogprob, ref top-5 | argmax same |\n|---|---|---|---|---|---|")
    for name, run in runs[1:]:
        rows = []
        for p in sorted(set(ref) & set(run)):
            a, b = ref[p], run[p]
            la, lb = logsm(a), logsm(b)
            rel = np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(a**2))
            kl = float(np.sum(np.exp(la) * (la - lb)))
            top = np.argsort(-a)[:5]
            dl = float(np.mean(np.abs(la[top] - lb[top])))
            same = int(np.argmax(a) == np.argmax(b))
            rows.append((rel, kl, dl, same))
            print(f"| {name} | {p} | {rel:.3e} | {kl:.4f} | {dl:.4f} | {'yes' if same else 'NO'} |")
        r = np.array(rows)
        print(f"| **{name}** | mean of {len(rows)} | {r[:,0].mean():.3e} | {r[:,1].mean():.4f} | {r[:,2].mean():.4f} | {int(r[:,3].sum())}/{len(rows)} |")


if __name__ == "__main__":
    main()
