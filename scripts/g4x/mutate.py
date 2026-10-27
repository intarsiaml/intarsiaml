#!/usr/bin/env python3
"""G4X planted mutants of the new/changed CUDA kernels: mutate.py <tree> <name> applies one in place; mutate.py --list prints
the names and the `kernels test` arguments (op, kernel) that must FAIL on the mutant. Each mutation is an exact one-place
string replacement; it fails (exit 2) when the text is not there exactly once, so a stale mutation cannot pass as 'caught'."""
import sys

SK = "kernels/cuda/attention_splitkv.cu"
FE = "kernels/cuda/attention_flash_ext.cu"
MG = "kernels/cuda/moe_grouped.cu"
MW = "kernels/cuda/moe_w4a16.cu"

# name: (file, old, new, kernel that must fail, op)
M = {
    # D = 512 split-KV decode / short prefill
    "sk512-pass": (SK, "g0 = (blockIdx.y % npass) * kGP512", "g0 = 0", "attn_decode_bf16_splitkv_ext", "attn_decode"),
    "sk512-hidim": (SK, "(e / 8) * 256 + c * 8 + (e % 8)]) * qscale;", "(e / 8) * 128 + c * 8 + (e % 8)]) * qscale;", "attn_decode_bf16_splitkv_ext", "attn_decode"),
    "sk512-merge": (SK, "for (int d = threadIdx.x; d < D; d += kBlock) {\n                float A = 0.f, L = 0.f;", "for (int d = threadIdx.x; d < 256; d += kBlock) {\n                float A = 0.f, L = 0.f;", "attn_decode_bf16_splitkv_ext", "attn_decode"),
    "sk512-window": (SK, "const int64_t lo = (window > 0 && visible > window) ? visible - window : 0;\n    int64_t start = (int64_t)sp * chunk;\n    int64_t end = start + chunk;\n    if (start < lo) start = lo;\n    if (end > visible) end = visible;\n\n    const ik_bf16* kb", "const int64_t lo = (window > 0 && visible > window) ? visible - window + 1 : 0;\n    int64_t start = (int64_t)sp * chunk;\n    int64_t end = start + chunk;\n    if (start < lo) start = lo;\n    if (end > visible) end = visible;\n\n    const ik_bf16* kb", "attn_decode_bf16_splitkv_ext", "attn_decode"),
    # tensor-core flash prefill, window and D = 512
    "fe-lowmask": (FE, "if (key >= lim || (window > 0 && key < lim - window)) v = -INFINITY;", "if (key >= lim) v = -INFINITY;", "attn_prefill_bf16_flash_ext", "attn_prefill"),
    "fe-lomax": (FE, "const int64_t lo_max = (window > 0 && kv_end > window) ? kv_end - window : 0;", "const int64_t lo_max = 0;", "attn_prefill_bf16_flash_ext", "attn_prefill"),
    "fe-vhalf": (FE, "const ik_bf16* vb = vc + (int64_t)kvh * cap * D + doff;", "const ik_bf16* vb = vc + (int64_t)kvh * cap * D;", "attn_prefill_bf16_flash_ext", "attn_prefill"),
    "fe-outhalf": (FE, "*reinterpret_cast<uint32_t*>(out + ((int64_t)row_hi * N + h) * D + doff + nb * 8 + 2 * c) =", "*reinterpret_cast<uint32_t*>(out + ((int64_t)row_hi * N + h) * D + nb * 8 + 2 * c) =", "attn_prefill_bf16_flash_ext", "attn_prefill"),
    # MoE half-unit tail (I = 64 mod 128)
    "moe-units": (MG, "const int U = (Kdim + 127) / 128;  // 128-wide raw units;", "const int U = Kdim / 128;  // 128-wide raw units;", "moe_experts_mma_gelu_bf16", "moe_experts"),
    "w4a16-units": (MW, "const int U = (Kdim + 127) / 128;  // 128-wide packed units;", "const int U = Kdim / 128;  // 128-wide packed units;", "moe_experts_w4a16_gelu_bf16", "moe_experts"),
    "w4a16-scale-tail": (MW, "cp_async4(rawu(slot) + tid * kRawStride + 68, second ? ssrc + u * 8 + 4 : ssrc, second ? 4 : 0);", "cp_async4(rawu(slot) + tid * kRawStride + 68, ssrc, 4);", "moe_experts_w4a16_gelu_bf16", "moe_experts"),
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
