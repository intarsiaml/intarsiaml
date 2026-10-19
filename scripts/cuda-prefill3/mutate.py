#!/usr/bin/env python3
"""Mutations of the matmul_fp8_gemm_reg kernel (kernels/cuda/fp8_gemm.cu, namespace reg) for the mutation check of
tests/fp8_gemm_reg.rs.

  mutate.py <tree-dir> <mutation>   apply (each `find` must occur exactly `count` times, else exit 1)
  mutate.py --list                  print the mutation names

Pattern of scripts/cuda-moe-w4a16/mutate.py: every mutation is a deliberate bug the tests must catch (cargo test exit 101),
except the ones marked EQUIVALENT, which do not change any stored output (listed so the report can name them).
"""
import sys

FILE = "kernels/cuda/fp8_gemm.cu"

# name -> (description, [(find, replace, count)])
MUTATIONS = {
    "sign-hi": (
        "the sign of the second bf16 of each pair is dropped",
        [("(y & 0x80008000u) | ((y >> 4)", "(y & 0x00008000u) | ((y >> 4)", 1)],
    ),
    "shift": (
        "exponent:mantissa bits shifted by 3 instead of 4",
        [("((y >> 4) & 0x07F007F0u)", "((y >> 3) & 0x07F007F0u)", 1)],
    ),
    "mant-mask": (
        "lowest mantissa bit of the e4m3 value dropped",
        [("((y >> 4) & 0x07F007F0u)", "((y >> 4) & 0x07E007E0u)", 1)],
    ),
    "exp-scale": (
        "bias restore multiplies by 2^119 in the high half",
        [("__ushort_as_bfloat16((unsigned short)0x7B80u));  // 2^120", "__ushort_as_bfloat16((unsigned short)0x7B00u));  // 2^120", 1)],
    ),
    "no-rebias": (
        "no multiply by 2^120 (values stay 2^-120 of the weight)",
        [("    v = __hmul2(v, k);\n", "", 1)],
    ),
    "byte-order": (
        "the two bytes of a pair swapped (k = 2t+1, 2t)",
        [("const uint32_t sel = 4u | (p << 4) | (4u << 8) | ((p + 1u) << 12);",
          "const uint32_t sel = 4u | ((p + 1u) << 4) | (4u << 8) | (p << 12);", 1)],
    ),
    "pair-select": (
        "odd lanes read the pair at byte 0 of the word instead of byte 2",
        [("const uint32_t p = 2u * (uint32_t)(t & 1);", "const uint32_t p = 0u;", 1)],
    ),
    "word-select": (
        "lanes t = 2, 3 read the first word of the 8-byte half",
        [("* LDW + 4 * (t >> 1);", "* LDW + 4 * (t >> 2);", 1)],
    ),
    "hi-offset": (
        "the second B register reads k + 4 instead of k + 8",
        [("bt + j * 8 * LDW + kk + 8)", "bt + j * 8 * LDW + kk + 4)", 1)],
    ),
    "b-swap": (
        "b0 and b1 swapped in the mma",
        [("ik_mma_bf16_16816(acc[i][j], a0, a1, a2, a3, b[j][0], b[j][1]);",
          "ik_mma_bf16_16816(acc[i][j], a0, a1, a2, a3, b[j][1], b[j][0]);", 1)],
    ),
    "b-row": (
        "B rows of n8 tiles 1..3 read 4 rows apart instead of 8",
        [("bt + j * 8 * LDW + kk)", "bt + j * 4 * LDW + kk)", 1), ("bt + j * 8 * LDW + kk + 8)", "bt + j * 4 * LDW + kk + 8)", 1)],
    ),
    "a-row": (
        "ldmatrix rows 8..15 of each m16 tile read rows 0..7",
        [("const int r = wm * 64 + i * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;", "const int r = wm * 64 + i * 16 + (lane & 7);", 1)],
    ),
    "warp-col": (
        "warp columns 16 apart instead of 32 (B fragments)",
        [("const int boff = (wn * 32 + (lane >> 2)) * LDW", "const int boff = (wn * 16 + (lane >> 2)) * LDW", 1)],
    ),
    "wait-depth": (
        "the ring waits for one group fewer (stage s may not have landed)",
        [("ik_cp_async_wait_group<STAGES - 2>();", "ik_cp_async_wait_group<STAGES - 1>();", 1)],
    ),
    "slot": (
        "stage s computed from the slot of stage s + 1",
        [("const ik_bf16* at = a_tile(s % STAGES);", "const ik_bf16* at = a_tile((s + 1) % STAGES);", 1)],
    ),
    "prefetch-dist": (
        "the next load goes one stage early, into the slot being read",
        [("if (s + STAGES - 1 < S) issue(s + STAGES - 1);", "if (s + STAGES - 2 < S) issue(s + STAGES - 2);", 1)],
    ),
    "w-chunk": (
        "weight copy chunks 8 bytes apart instead of 16",
        [("wsrc[q] = w + (int64_t)(wv[q] ? n0 + r : 0) * K + (tid & 3) * 16;", "wsrc[q] = w + (int64_t)(wv[q] ? n0 + r : 0) * K + (tid & 3) * 8;", 1)],
    ),
    "w-tail": (
        "weight rows past N are copied (row 0 instead of zeros) and N-tail rows are cut at N - 1",
        [("wv[q] = n0 + r < N;", "wv[q] = n0 + r < N - 1;", 1)],
    ),
    "a-tail": (
        "EQUIVALENT: activation rows past M are copied (row 0) instead of zero-filled (they feed only output rows past M,"
        " which are never stored)",
        [("ik_cp_async16(a + ((tid >> 3) + (NT / 8) * q) * LDA + (tid & 7) * 8, asrc[q] + (int64_t)st * BK, av[q] ? 16 : 0);",
          "ik_cp_async16(a + ((tid >> 3) + (NT / 8) * q) * LDA + (tid & 7) * 8, asrc[q] + (int64_t)st * BK, 16);", 1)],
    ),
    "act-src": (
        "W8A8 reads the raw activations instead of the e4m3-quantised ones",
        [("const ik_bf16* const a_src = actq ? xq : x;\n    const float s_out = gs[0] * (actq ? gs[1] : 1.f);\n    auto a_tile",
          "const ik_bf16* const a_src = x;\n    const float s_out = gs[0] * (actq ? gs[1] : 1.f);\n    auto a_tile", 1)],
    ),
    "out-scale": (
        "the input scale is not applied in the epilogue",
        [("const float s_out = gs[0] * (actq ? gs[1] : 1.f);\n    auto a_tile", "const float s_out = gs[0];\n    auto a_tile", 1)],
    ),
    "tile-order": (
        "block -> tile map: the M index uses the other order",
        [("const int m0 = (nfast ? bid / nnt : bid % nmt) * BM;", "const int m0 = (nfast ? bid % nmt : bid / nnt) * BM;", 1)],
    ),
    "vec-col": (
        "vectorised epilogue: columns of the second n8 tile half shifted",
        [("const int cl = wn * 32 + j * 8 + (lane & 3) * 2;", "const int cl = wn * 32 + j * 8 + (lane & 3);", 1)],
    ),
    "scalar-col": (
        "scalar epilogue: the second column of a pair is written to the first",
        [("if (col + 1 < N) ik_store(out + (int64_t)row * N + col + 1, acc[i][j][2 * hh + 1] * s_out);\n                }\n    }\n}",
          "if (col + 1 < N) ik_store(out + (int64_t)row * N + col, acc[i][j][2 * hh + 1] * s_out);\n                }\n    }\n}", 1)],
    ),
    "fallback-off": (
        "EQUIVALENT: the small-grid fallback to the 64-row kernel never taken (same bits, other schedule)",
        [("    if (tiles < 24) {", "    if (tiles < 0) {", 1)],
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
