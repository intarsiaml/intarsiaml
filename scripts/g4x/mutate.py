#!/usr/bin/env python3
"""G4X planted mutants of the new D = 512 split-KV attention kernel (the prefill kernels are on lane-g4x/gemma4-prefill-ref): mutate.py <tree> <name> applies one in place; mutate.py --list prints
the names and the `kernels test` arguments (op, kernel) that must FAIL on the mutant. Each mutation is an exact one-place
string replacement; it fails (exit 2) when the text is not there exactly once, so a stale mutation cannot pass as 'caught'."""
import sys

SK = "kernels/cuda/attention_splitkv.cu"

# name: (file, old, new, kernel that must fail, op)
M = {
    # D = 512 split-KV decode / short prefill
    "sk512-pass": (SK, "g0 = (blockIdx.y % npass) * kGP512", "g0 = 0", "attn_decode_bf16_splitkv_ext", "attn_decode"),
    "sk512-hidim": (SK, "(e / 8) * 256 + c * 8 + (e % 8)]) * qscale;", "(e / 8) * 128 + c * 8 + (e % 8)]) * qscale;", "attn_decode_bf16_splitkv_ext", "attn_decode"),
    "sk512-merge": (SK, "for (int d = threadIdx.x; d < D; d += kBlock) {\n                float A = 0.f, L = 0.f;", "for (int d = threadIdx.x; d < 256; d += kBlock) {\n                float A = 0.f, L = 0.f;", "attn_decode_bf16_splitkv_ext", "attn_decode"),
    "sk512-window": (SK, "const int64_t lo = (window > 0 && visible > window) ? visible - window : 0;\n    int64_t start = (int64_t)sp * chunk;\n    int64_t end = start + chunk;\n    if (start < lo) start = lo;\n    if (end > visible) end = visible;\n\n    const ik_bf16* kb", "const int64_t lo = (window > 0 && visible > window) ? visible - window + 1 : 0;\n    int64_t start = (int64_t)sp * chunk;\n    int64_t end = start + chunk;\n    if (start < lo) start = lo;\n    if (end > visible) end = visible;\n\n    const ik_bf16* kb", "attn_decode_bf16_splitkv_ext", "attn_decode"),

}

if __name__ == "__main__":
    if sys.argv[1] == "--list":
        for k, v in M.items():
            print(k, v[3], v[4])
        sys.exit(0)
    tree, name = sys.argv[1], sys.argv[2]
    f, old, new, _, _ = M[name]
    p = f"{tree}/{f}"
    s = open(p).read()
    if s.count(old) != 1:
        print(f"{name}: expected exactly one match in {f}, found {s.count(old)}")
        sys.exit(2)
    open(p, "w").write(s.replace(old, new))
    print(f"{name}: applied to {f}")
