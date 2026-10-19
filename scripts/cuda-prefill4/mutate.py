#!/usr/bin/env python3
"""Mutations of the gdn_delta_rule_bf16_split kernel (kernels/cuda/gdn.cu, gdn_delta_split_kernel and its launcher) for
the mutation check of tests/gdn_split.rs.

  mutate.py <tree-dir> <mutation>   apply (each `find` must occur exactly `count` times, else exit 1)
  mutate.py --list                  print the mutation names

Pattern of scripts/cuda-prefill3/mutate.py: every mutation is a deliberate bug the tests must catch (cargo test exit 101),
except the ones marked EQUIVALENT, which cannot change any stored bit (listed so the report can name them).
"""
import sys

FILE = "kernels/cuda/gdn.cu"

# name -> (description, [(find, replace, count)])
MUTATIONS = {
    "kv-group-order": (
        "kv group sum contracted the other way (FMUL x, FFMA y instead of FMUL y, FFMA x)",
        [("__fmaf_rn(s[i].x, kk4.x, __fmul_rn(s[i].y, kk4.y))", "__fmaf_rn(s[i].y, kk4.y, __fmul_rn(s[i].x, kk4.x))", 1)],
    ),
    "out-group-order": (
        "output group sum contracted the other way",
        [("__fmaf_rn(s[i].x, qq4.x, __fmul_rn(s[i].y, qq4.y))", "__fmaf_rn(s[i].y, qq4.y, __fmul_rn(s[i].x, qq4.x))", 1)],
    ),
    "kv-carry-order": (
        "kv carry handed over in reverse lane order (groups summed in another order)",
        [("kv = __shfl_sync(0xffffffffu, c, row0 + st);", "kv = __shfl_sync(0xffffffffu, c, row0 + LPR - 1 - st);", 1)],
    ),
    "out-carry-order": (
        "output carry handed over in reverse lane order",
        [("out = st + 1 < LPR ? __shfl_sync(0xffffffffu, c, row0 + st) : c;",
          "out = st + 1 < LPR ? __shfl_sync(0xffffffffu, c, row0 + LPR - 1 - st) : c;", 1)],
    ),
    "group-chain-tree": (
        "a lane's groups added as two halves (a tree) instead of left to right, in both dot products",
        [("            for (int i = 0; i < NG; ++i) c = __fadd_rn(c, g[i]);",
          "            {\n                float lo = 0.f, hi = 0.f;\n                for (int i = 0; i < NG / 2; ++i) lo = __fadd_rn(lo, g[i]);\n"
          "                for (int i = NG / 2; i < NG; ++i) hi = __fadd_rn(hi, g[i]);\n                c = __fadd_rn(c, __fadd_rn(lo, hi));\n            }", 2)],
    ),
    "delta-sign": (
        "delta = (kv - v) * beta",
        [("__fsub_rn(v, kv)", "__fsub_rn(kv, v)", 1)],
    ),
    "decay-zw": (
        "the decay is not applied to the z / w components of the state",
        [("            s[i].z = __fmul_rn(s[i].z, decay); s[i].w = __fmul_rn(s[i].w, decay);\n", "", 1)],
    ),
    "update-uses-q": (
        "the x component of the update uses q instead of k",
        [("s[i].x = __fmaf_rn(kk4.x, delta, s[i].x);", "s[i].x = __fmaf_rn(qq4.x, delta, s[i].x);", 1)],
    ),
    "out-lane": (
        "the output is written by the first lane of a row (which holds a partial sum only)",
        [("if (valid && l == LPR - 1)", "if (valid && l == 0)", 1)],
    ),
    "tile-pad": (
        "the state tile slot of the coalesced load / store ignores the pad float4 (disagrees with the lane's own slots)",
        [("return rr * (DK / 4 + LPR) + (gi / NG) * (NG + 1) + gi % NG;", "return rr * (DK / 4 + LPR) + (gi / NG) * NG + gi % NG;", 1)],
    ),
    "tile-mine": (
        "a lane reads / writes its own state slots without the pad float4",
        [("const int mine = r * (DK / 4 + LPR) + l * (NG + 1);", "const int mine = r * (DK / 4 + LPR) + l * NG;", 1)],
    ),
    "stage-pad": (
        "q / k staged without the pad float4 (the reads expect it)",
        [("const int at = 4 * (g + g / NG) + (e & 3);", "const int at = 4 * g + (e & 3);", 1)],
    ),
    "stage-first": (
        "a lane reads its staged q / k groups without the pad float4",
        [("const int first = l * (NG + 1);", "const int first = l * NG;", 1)],
    ),
    "prefetch-index": (
        "the raw inputs prefetched for token t + 1 instead of t + 2 (the next token's norm sees the wrong token)",
        [("load_raw(t + 2 < T ? t + 2 : T - 1);", "load_raw(t + 1 < T ? t + 1 : T - 1);", 1)],
    ),
    "prep-buffer": (
        "the next token's q / k written into the current buffer",
        [("prep(cur ^ 1);", "prep(cur);", 1)],
    ),
    "prologue-prefetch": (
        "the prologue prefetches token 0 again instead of token 1",
        [("load_raw(T > 1 ? 1 : 0);", "load_raw(0);", 1)],
    ),
    "store-guard": (
        "the coalesced store writes every row of the last block, past dv (into the next head's rows)",
        [("if (q / (DK / 4) < nrows) blk[q]", "blk[q]", 1)],
    ),
    "nrows": (
        "a block always takes RPW rows (the partial last block of a head reads / writes the next head's rows)",
        [("const int nrows = min(RPW, dv - (int)blockIdx.x * RPW);", "const int nrows = RPW;", 1)],
    ),
    "fast-exp": (
        "the decay uses __expf (fast approximate exp) instead of expf",
        [("ndecay = expf(neg_a * softplus(araw + dtb));", "ndecay = __expf(neg_a * softplus(araw + dtb));", 1)],
    ),
    "q-norm": (
        "q staged with k's norm",
        [("sq[buf][at] = qr_[i] * qn;", "sq[buf][at] = qr_[i] * kn;", 1)],
    ),
    "v-zero": (
        "v is zero",
        [("nv = vraw;", "nv = 0.f;", 1)],
    ),
    "beta-one": (
        "beta = 1 (the sigmoid of b dropped)",
        [("nbeta = sigmoidf(braw);", "nbeta = 1.f;", 1)],
    ),
    "no-loop-sync": (
        "no __syncwarp at the top of a token (the double buffer is unordered; a race that may go unseen)",
        [("        __syncwarp();  // buffer `cur`", "        // __syncwarp();  // buffer `cur`", 1)],
    ),
    "EQ-add-commute": (
        "EQUIVALENT: c + g written as g + c (IEEE addition is commutative: same bits)",
        [("c = __fadd_rn(c, g[i]);", "c = __fadd_rn(g[i], c);", 2)],
    ),
    "EQ-last-shuffle": (
        "EQUIVALENT: the output's last stage also shuffles (from lane LPR - 1 itself: the writer already holds the value)",
        [("out = st + 1 < LPR ? __shfl_sync(0xffffffffu, c, row0 + st) : c;", "out = __shfl_sync(0xffffffffu, c, row0 + st);", 1)],
    ),
    "EQ-lpr-default": (
        "EQUIVALENT: 4 lanes per row by default instead of 2 (the bits do not depend on it; slower)",
        [("return (n == 2 || n == 4 || n == 8) ? n : 2;", "return (n == 2 || n == 4 || n == 8) ? n : 4;", 1)],
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
