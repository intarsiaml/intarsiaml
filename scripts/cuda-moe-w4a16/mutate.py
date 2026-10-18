#!/usr/bin/env python3
"""Mutations of kernels/cuda/moe_w4a16.cu for the mutation check of tests/moe_w4a16.rs.

  mutate.py <tree-dir> <mutation>   apply (each `find` must occur exactly `count` times, else exit 1)
  mutate.py --list                  print the mutation names

Pattern of scripts/cuda-moe-nvfp4/mutate.py: every mutation is a deliberate bug the tests must catch (cargo test exit 101).
"""
import sys

FILE = "kernels/cuda/moe_w4a16.cu"

# name -> (description, [(find, replace, count)])
MUTATIONS = {
    "sign-lo": (
        "sign bit of elements 0 and 1 dropped (r0 = magnitudes only)",
        [(" | (((q & 0x88u) * 0x1001000u) & 0x80008000u);", ";", 1)],
    ),
    "sign-hi-shift": (
        "sign bit of element 2 moved to the wrong position",
        [("(q & 0x8800u) * 0x10010u", "(q & 0x8800u) * 0x10020u", 1)],
    ),
    "lut-hi": (
        "bf16 high byte of the magnitude 2.0 is 0 (LUT entry 4)",
        [("0x40404040u, sel", "0x40404000u, sel", 1)],
    ),
    "lut-lo": (
        "bf16 low byte of the magnitude 3.0 is 0x80 (LUT entry 5)",
        [("0xC0804000u, sel", "0xC0808000u, sel", 1)],
    ),
    "interleave": (
        "low and high bytes of the first bf16 pair swapped",
        [("__byte_perm(lo, hi, 0x5140u)", "__byte_perm(hi, lo, 0x5140u)", 1)],
    ),
    "byte-select": (
        "the second packed word is read at byte t+1 instead of byte t",
        [("((uint32_t)(4 + t4) << 4)", "((uint32_t)(5 + t4) << 4)", 1)],
    ),
    "scale-byte": (
        "k16 blocks 0/1 and 2/3 of a step use each other's block scale",
        [("(scs[j] >> (8 * kk)) & 0xFFu", "(scs[j] >> (8 * (kk ^ 1))) & 0xFFu", 1)],
    ),
    "scale-step": (
        "the second k64 step of a unit reuses the scales of the first",
        [("kRawStride + 64 + (s & 1) * 4;", "kRawStride + 64;", 1)],
    ),
    "raw-slot": (
        "the packed ring slot of the odd steps is off by one unit",
        [("const unsigned char* rb = rawu((s >> 1) % NR) + (s & 1) * 32;", "const unsigned char* rb = rawu(((s + 1) >> 1) % NR) + (s & 1) * 32;", 1)],
    ),
    "raw-ring": (
        "the packed ring has one slot (the next unit overwrites the one in use)",
        [("NR = LR / 2 + 1;", "NR = LR / 2;", 1)],
    ),
    "a-ring": (
        "the A ring is one slot short",
        [("constexpr int NA = LA + 1, NR", "constexpr int NA = LA, NR", 1)],
    ),
    "accumulate-in-mma": (
        "k16 products accumulated inside the mma instead of formed from zero and added (the original accuracy bug)",
        [("ik_mma_bf16_16816(tt, a0, a1, a2, a3, b[j][0], b[j][1]);",
          "ik_mma_bf16_16816(acc[i][j], a0, a1, a2, a3, b[j][0], b[j][1]); tt[0] = tt[1] = tt[2] = tt[3] = 0.f;", 1)],
    ),
    "skip-partial": (
        "the m16 tile that is only partly filled is skipped",
        [("if (i * 16 >= rows) continue;", "if ((i + 1) * 16 > rows) continue;", 1)],
    ),
    "a-col": (
        "A chunk columns swapped in pairs",
        [("arow[q] + st * BK + c8 * 8", "arow[q] + st * BK + (c8 ^ 1) * 8", 1)],
    ),
    "raw-unit": (
        "packed weights of unit u read from unit u^1",
        [("rsrc[q] + u * 64", "rsrc[q] + (u ^ 1) * 64", 1)],
    ),
    "scale-unit": (
        "scales of unit u read from unit u^1",
        [("ssrc + u * 8", "ssrc + (u ^ 1) * 8", 1)],
    ),
    "up-rows": (
        "up projection computed from the gate rows",
        [("(r < R / 2 ? n0 + r : I + n0 + (r - R / 2))", "(r < R / 2 ? n0 + r : n0 + (r - R / 2))", 1)],
    ),
    "up-uses-gate-scale": (
        "up projection multiplied by the gate's global scale",
        [("acc[i][j + JH][2 * hh + v] * gu", "acc[i][j + JH][2 * hh + v] * gg", 1)],
    ),
    "silu-sign": (
        "SiLU with the wrong sign: g / (1 + exp(g))",
        [("expf(-gt)", "expf(gt)", 1)],
    ),
    "gather": (
        "token index of a pair computed with the wrong divisor",
        [("pairs[begin + m0 + r] / K", "pairs[begin + m0 + r] / (K + 1)", 1)],
    ),
    "scatter": (
        "down output written in sorted order instead of scattered to the pair's (token, slot) row",
        [("else dst = out + (int64_t)pairs[begin + m0 + rl] * H + n0 + part * 8;",
          "else dst = out + (int64_t)(begin + m0 + rl) * H + n0 + part * 8;", 1)],
    ),
    "valid-rows": (
        "one row past the tile's last pair is written as well",
        [("if (rl >= rows) continue;\n        ik_bf16* dst;", "if (rl > rows) continue;\n        ik_bf16* dst;", 1)],
    ),
    "frag-col": (
        "gate/up epilogue column offset of the lane (2 * t4) replaced by t4",
        [("const int cl = wn * 8 * JH + j * 8 + t4 * 2;", "const int cl = wn * 8 * JH + j * 8 + t4;", 1)],
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
