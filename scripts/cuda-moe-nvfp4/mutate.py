#!/usr/bin/env python3
"""Mutations of kernels/cuda/moe_nvfp4_mma_sm120a.cu for the mutation check of tests/moe_nvfp4.rs.

  mutate.py <tree-dir> <mutation>   apply (each `find` must occur exactly `count` times, else exit 1)
  mutate.py --list                  print the mutation names

Pattern of scripts/cuda-decode/mutate.py: every mutation is a deliberate bug the tests must catch (cargo test exit 101).
"""
import sys

FILE = "kernels/cuda/moe_nvfp4_mma_sm120a.cu"

# name -> (description, [(find, replace, count)])
MUTATIONS = {
    "sfa-rows": (
        "A scales of rows g+8 taken from row g (lane t == 1 supplies the wrong row)",
        [("(r0 + 8 * t) * 8 + st * 4", "r0 * 8 + st * 4", 1)],
    ),
    "sfa-lane": (
        "only lane t == 0 supplies A scales (rows g+8 get zero scales)",
        [("const uint32_t sfa = t < 2 ?", "const uint32_t sfa = t < 1 ?", 1)],
    ),
    "sfb-step": (
        "B scales of the second k64 step of a unit read from the first",
        [("ws + row * 8 + st * 4", "ws + row * 8", 1)],
    ),
    "b-second-half": (
        "B fragment register b1 read from the first 16 bytes of the step (k 32..63 replaced by k 0..31 again)",
        [("wr + row * kRawStride + st * 32 + 16 + 4 * t", "wr + row * kRawStride + st * 32 + 4 * t", 1)],
    ),
    "no-ginv": (
        "epilogue does not undo the per-row activation scale",
        [("            ri[i][hh] = v;", "            ri[i][hh] = rl < rows ? 1.f : 0.f;", 1)],
    ),
    "up-uses-gate-scale": (
        "up projection multiplied by the gate's global scale",
        [("(ri[i][hh] * gu)", "(ri[i][hh] * gg)", 1)],
    ),
    "silu-sign": (
        "SiLU with the wrong sign: g / (1 + exp(g))",
        [("gt / (1.f + expf(-gt))", "gt / (1.f + expf(gt))", 1)],
    ),
    "quant-scale": (
        "activation block scale from amax / 7 instead of amax / 6",
        [("(bmax * gx) * (1.f / 6.f)", "(bmax * gx) * (1.f / 7.f)", 1)],
    ),
    "quant-nibbles": (
        "e2m1 pairs packed high nibble first (the two elements of every byte swapped)",
        [('"cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\\n"', '"cvt.rn.satfinite.e2m1x2.f32 b0, %1, %2;\\n"', 1)],
    ),
    "quant-no-sat-guard": (
        "inverse block scale computed with the unrounded scale (gx / (amax * gx / 6)) instead of the e4m3 one",
        [("__fdiv_rn(gx, sd)", "__fdiv_rn(6.f, bmax > 0.f ? bmax : 1.f)", 1)],
    ),
    "a-unit-offset": (
        "activation bytes of unit u read from u * 32 instead of u * 64",
        [("aq + arow * (Kdim / 2) + u * 64 + rpart * 16", "aq + arow * (Kdim / 2) + u * 32 + rpart * 16", 1)],
    ),
    "valid-rows": (
        "rows past the end of an expert's pair list are read as real pairs (tile tail not masked)",
        [("a_ok = ar < rows;", "a_ok = ar < BM;", 1)],
    ),
    "up-rows": (
        "up projection rows taken from the gate rows",
        [("I + n0 + (r - 64)", "n0 + (r - 64)", 1)],
    ),
    "scatter": (
        "down output written in sorted order instead of scattered to the pair's (token, slot) row",
        [("else dst = out + (int64_t)pairs[begin + m0 + rl] * H + n0 + part * 8;",
          "else dst = out + (int64_t)(begin + m0 + rl) * H + n0 + part * 8;", 1)],
    ),
}


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "--list":
        print(" ".join(MUTATIONS))
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


if __name__ == "__main__":
    sys.exit(main())
