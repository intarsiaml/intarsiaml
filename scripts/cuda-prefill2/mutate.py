#!/usr/bin/env python3
"""Mutations of kernels/cuda/gdn_chunked.cu for the mutation check of tests/gdn_chunked.rs.

  mutate.py <tree-dir> <mutation>     apply (each `find` must occur exactly as often as listed, else exit 1)
  mutate.py --list                    print the mutation names

Pattern of scripts/cuda-decode/mutate.py: every mutation is a deliberate bug the tests must catch (cargo test exit 101).
"""
import sys

FILE = "kernels/cuda/gdn_chunked.cu"

# name -> (description, [(find, replace, count)])
MUTATIONS = {
    "no-interdecay": (
        "state update without the inter-chunk decay e_last * S0",
        [("sp[0] = e_last * sp[0] + acc[i][0];", "sp[0] = sp[0] + acc[i][0];", 1),
         ("sp[1] = e_last * sp[1] + acc[i][1];", "sp[1] = sp[1] + acc[i][1];", 1)],
    ),
    "no-kd": (
        "state update k scale without the (e_last / e_r) factor",
        [("aux[4 * kC + tid] = rk * expf(Glast - G);", "aux[4 * kC + tid] = rk;", 1)],
    ),
    "mask-strict": (
        "causal mask of M strict (p < r): the diagonal term q_r.k_r delta_r is dropped",
        [("sB[idx] = p <= r ? ex * gqv : 0.f;", "sB[idx] = p < r ? ex * gqv : 0.f;", 1)],
    ),
    "no-solve": (
        "skip the triangular-solve correction: T = I",
        [("sT[r * kC + p] = -acc;", "sT[r * kC + p] = 0.f;", 1)],
    ),
    "no-carry": (
        "no state carry between chunks: S = delta-sum only",
        [("sp[0] = e_last * sp[0] + acc[i][0];", "sp[0] = acc[i][0];", 1),
         ("sp[1] = e_last * sp[1] + acc[i][1];", "sp[1] = acc[i][1];", 1)],
    ),
    "tail-decay": (
        "padded tokens of a tail chunk decay the state (g = -0.5 instead of 0)",
        [("float g = 0.f, bt = 0.f;", "float g = -0.5f, bt = 0.f;", 1)],
    ),
    "no-intra-decay": (
        "no e_r / e_p decay inside L",
        [("sA[idx] = p < r ? sBG[r] * ex * gkv : 0.f;", "sA[idx] = p < r ? sBG[r] * gkv : 0.f;", 1)],
    ),
    "no-qe": (
        "q scale without the e_r factor (the S0 term of the output)",
        [("aux[3 * kC + tid] = rq * e;", "aux[3 * kC + tid] = rq;", 1)],
    ),
    "no-qscale": (
        "q without the 1/sqrt(dk) scale",
        [("rsqrtf(qq + eps) * scale : 0.f;", "rsqrtf(qq + eps) : 0.f;", 1)],
    ),
}

ARGS = {name: "--test gdn_chunked" for name in MUTATIONS}


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--list":
        print(" ".join(MUTATIONS))
        return 0
    if len(sys.argv) == 3 and sys.argv[1] == "--args":
        print(ARGS[sys.argv[2]])
        return 0
    if len(sys.argv) != 3 or sys.argv[2] not in MUTATIONS:
        print(__doc__)
        return 2
    root, name = sys.argv[1], sys.argv[2]
    path = f"{root}/{FILE}"
    src = open(path).read()
    for find, repl, count in MUTATIONS[name][1]:
        if src.count(find) != count:
            print(f"mutation {name}: `{find}` occurs {src.count(find)} times, expected {count}")
            return 1
        src = src.replace(find, repl)
    open(path, "w").write(src)
    print(f"applied {name}: {MUTATIONS[name][0]}")
    return 0


sys.exit(main())
