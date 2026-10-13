#!/usr/bin/env python3
"""Apply one named mutation to a copy of the tree (usage: mutate.py <tree> <name>). Used by `remote.sh mutate`:
each mutation must make `cargo test -p intarsia-backend-hip --test decode_v2` FAIL, which shows the kernel-level tests
catch a wrong accumulation, a skipped iteration, a flipped tie-break, or a dropped state / tail write."""
import sys

MUTATIONS = {
    # fused single-tile conv: the thread's new state is written doubled
    "conv-state": ("kernels/hip/gdn.hip", [(
        "        for (int j = 0; j < K - 1; ++j) state[(int64_t)c * (K - 1) + j] = h[j];\n}\n\n// state[c] =",
        "        for (int j = 0; j < K - 1; ++j) state[(int64_t)c * (K - 1) + j] = h[j] * 2.f;\n}\n\n// state[c] =")]),
    # nvfp4 gemv v2: the first lane-iteration after the prefetch window is skipped (only visible for K > 2048)
    "nvfp4-iteration": ("kernels/hip/nvfp4.hip", [(
        "for (int u = lane + IK_WARP * kPre; u < nu; u += IK_WARP) {  // K > 2048",
        "for (int u = lane + IK_WARP * (kPre + 1); u < nu; u += IK_WARP) {  // K > 2048")]),
    # moe top-k v2: ties go to the highest expert index instead of the lowest
    "topk-tiebreak": ("kernels/hip/moe.hip", [
        ("(topk_key16(b.bits) << 16) | (0xffffu - (uint32_t)e)", "(topk_key16(b.bits) << 16) | (uint32_t)e"),
        ("const int bi = (int)(0xffffu - (best & 0xffffu));", "const int bi = (int)(best & 0xffffu);")]),
    # rope: the in-kernel copy of the un-rotated tail skips every other value
    "rope-tail": ("kernels/hip/elementwise.hip", [(
        "for (int m = RD + k; m < D; m += half) out[base + m] = x[base + m];",
        "for (int m = RD + k; m < D; m += 2 * half) out[base + m] = x[base + m];")]),
    # moe experts v2 down kernel: the first half of the intermediate row is replaced by its second half
    "moe-down": ("kernels/hip/moe.hip", [(
        "                step(k, wv[k], sb[k], hv[XS ? 0 : k][0], hv[XS ? 0 : k][1]);",
        "                step(k, wv[k], sb[k], hv[XS ? 0 : k][1], hv[XS ? 0 : k][1]);")]),
    # fp8 gemv v2 (default staging Q=2): the output scale of the 2^-8-scaled hardware conversion is halved
    "fp8-scale": ("kernels/hip/fp8.hip", [(
        "((Q == 2 || Q == 3) ? 65536.f : 256.f)", "((Q == 2 || Q == 3) ? 32768.f : 256.f)")]),
    # fp8 gemv v2: the upper two bytes of each converted e4m3 pair read the lower pair's selector
    "fp8-convert": ("kernels/hip/fp8.hip", [(
        "e4m3x2_to_f16x2((uint32_t)c0, 0x030C020Cu)", "e4m3x2_to_f16x2((uint32_t)c0, 0x010C000Cu)")]),
}

tree, name = sys.argv[1], sys.argv[2]
path, pairs = MUTATIONS[name]
p = f"{tree}/{path}"
s = open(p).read()
for old, new in pairs:
    if s.count(old) != 1:
        sys.exit(f"mutation {name}: pattern not found exactly once in {path}: {old[:60]!r} ({s.count(old)})")
    s = s.replace(old, new)
open(p, "w").write(s)
print(f"applied {name} to {p}")
