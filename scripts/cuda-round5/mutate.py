#!/usr/bin/env python3
"""Mutations of the fp8 KV-cache attention kernels (attn_*_f8kv_*: the e4m3 paths of kernels/cuda/attention_splitkv.cu
and kernels/cuda/attention_flash.cu) for the mutation check of tests/attn_f8kv.rs.

  mutate.py <tree-dir> <mutation>   apply (each `find` must occur exactly `count` times, else exit 1)
  mutate.py --list [file]           print the mutation names (only those of kernels/cuda/<file> if given)

Pattern of scripts/cuda-prefill4/mutate.py. Only fp8-specific code is mutated: the bf16 kernels are the reference the
tests compare with (a mutation of shared code would change both sides alike). A race-type mutant (a tile converted after
the barrier that publishes it) is for the race check (scripts/cuda-checks/racecheck.sh) if the tests miss it.
"""
import sys

SK = "kernels/cuda/attention_splitkv.cu"
FL = "kernels/cuda/attention_flash.cu"

# name -> (file, description, [(find, replace, count)])
MUTATIONS = {
    # ---- split-KV (decode, <= 16 rows) ----
    "sk-unpack-swap": (SK, "e4m3 pairs unpacked in swapped order (dims 0/1 exchanged)",
        [("            f[4 * i + 0] = a.x;\n            f[4 * i + 1] = a.y;\n            f[4 * i + 2] = b.x;",
          "            f[4 * i + 0] = a.y;\n            f[4 * i + 1] = a.x;\n            f[4 * i + 2] = b.x;", 1)]),
    "sk-unpack-hi": (SK, "the high pair of each word decoded from the low half",
        [("            const float2 b = ik_e4m3x2_to_f2(static_cast<unsigned short>(w[i] >> 16));\n            f[4 * i + 0]",
          "            const float2 b = ik_e4m3x2_to_f2(static_cast<unsigned short>(w[i] & 0xFFFFu));\n            f[4 * i + 0]", 1)]),
    "sk-zero-nan": (SK, "keys that are not loaded hold e4m3 NaN instead of 0 (0 * NaN poisons the V sum)",
        [("    static __device__ __forceinline__ V zero() { return make_uint2(0, 0); }",
          "    static __device__ __forceinline__ V zero() { return make_uint2(0x7f7f7f7fu, 0x7f7f7f7fu); }", 1)]),
    "sk-load-narrow": (SK, "only 4 of a thread's 8 bytes are loaded",
        [("    static __device__ __forceinline__ V load(const uint8_t* p) { return __ldg(reinterpret_cast<const uint2*>(p)); }",
          "    static __device__ __forceinline__ V load(const uint8_t* p) { return make_uint2(__ldg(reinterpret_cast<const uint32_t*>(p)), 0u); }", 1)]),
    "sk-dispatch-d": (SK, "the fp8 path dispatches head_dim 128 with the head_dim-64 lane count",
        [("            case 128: rc = dispatch_group<16>((int)(N / NKV), qp, kp, vp, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, pos_dev, s); break;\n            case 256: rc = dispatch_group<32>((int)(N / NKV), qp, kp, vp, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, pos_dev, s); break;\n        }\n    }",
          "            case 128: rc = dispatch_group<8>((int)(N / NKV), qp, kp, vp, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, pos_dev, s); break;\n            case 256: rc = dispatch_group<32>((int)(N / NKV), qp, kp, vp, pacc, pml, T, N, NKV, cap, pos0, qscale, nsplit, pos_dev, s); break;\n        }\n    }", 1)]),
    "sk-decode-flag": (SK, "the fp8 decode launcher takes the prefill grid policy (decode = false)",
        [("    return splitkv_launch(q, kc, vc, pos0, scale, out, scratch, pos, stream, true, true);",
          "    return splitkv_launch(q, kc, vc, pos0, scale, out, scratch, pos, stream, false, true);", 1)]),
    # ---- flash (prefill chunks > 16 rows) ----
    "fl-no-land-k": (FL, "K tiles are never converted into the bf16 tile (stale / zero K)",
        [("        if constexpr (F8) land8(Ks, Kst);\n", "", 1)]),
    "fl-no-land-v": (FL, "V tiles are never converted into the bf16 tile",
        [("        if constexpr (F8) land8(Vs, Vst);\n", "", 1)]),
    "fl-land-v-after-sync": (FL, "RACE: V is converted after the barrier that publishes the tile (other warps may read it first)",
        [("        if constexpr (F8) land8(Vs, Vst);\n        __syncthreads();",
          "        __syncthreads();\n        if constexpr (F8) land8(Vs, Vst);", 1)]),
    "fl-land-v-before-wait": (FL, "V staging is read before cp.async.wait_group (bytes may not have landed)",
        [("        cp_wait<1>();     // V[kt] landed (the K[kt+1] group may still be in flight)\n        if constexpr (F8) land8(Vs, Vst);",
          "        if constexpr (F8) land8(Vs, Vst);\n        cp_wait<1>();     // V[kt] landed (the K[kt+1] group may still be in flight)", 1)]),
    "fl-src-offset": (FL, "a 16-byte copy reads the source at the bf16 chunk offset (ch * 8 bytes)",
        [("            cp_async16(st + r * D + ch * 16, valid ? src + key * D + ch * 16 : src, valid);",
          "            cp_async16(st + r * D + ch * 16, valid ? src + key * D + ch * 8 : src, valid);", 1)]),
    "fl-half-swap": (FL, "the two 8-dim halves of a 16-byte group land in each other's place",
        [("            *reinterpret_cast<uint4*>(dst + r * LD + ch * 16) = e4m3x8_to_bf16x8(v[i].x, v[i].y);\n            *reinterpret_cast<uint4*>(dst + r * LD + ch * 16 + 8) = e4m3x8_to_bf16x8(v[i].z, v[i].w);",
          "            *reinterpret_cast<uint4*>(dst + r * LD + ch * 16) = e4m3x8_to_bf16x8(v[i].z, v[i].w);\n            *reinterpret_cast<uint4*>(dst + r * LD + ch * 16 + 8) = e4m3x8_to_bf16x8(v[i].x, v[i].y);", 1)]),
    "fl-shared-staging": (FL, "V is staged in the K tile (the K[kt+1] copy overwrites V[kt]'s bytes before they are converted)",
        [("    unsigned char* Vst = reinterpret_cast<unsigned char*>(Vs) + ST_OFF;", "    unsigned char* Vst = reinterpret_cast<unsigned char*>(Ks) + ST_OFF;", 1)]),
    "fl-read-past-end": (FL, "rows past the visible keys are copied from the cache instead of zero-filled",
        [("            cp_async16(st + r * D + ch * 16, valid ? src + key * D + ch * 16 : src, valid);",
          "            cp_async16(st + r * D + ch * 16, src + key * D + ch * 16, key < cap);", 1)]),
    "fl-land-no-sync": (FL, "RACE: no barrier between reading the staged bytes and writing the bf16 tile over them",
        [("        __syncthreads();  // f8: every thread holds its staged bytes\n", "", 1)]),
    # classified after the run (it survived): land8 reads every thread's staged bytes before its barrier and writes the bf16
    # tile after it, so any place inside the (free) tile holds the staging bytes equally well
    "EQ-staging-low": (FL, "EQUIVALENT: the staging bytes sit at the start of the tile (land8's barrier separates all reads from all writes)",
        [("    constexpr int ST_OFF = BK * LD * 2 - BK * D;", "    constexpr int ST_OFF = 0;", 1)]),
}


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--list":
        f = sys.argv[2] if len(sys.argv) > 2 else None
        print(" ".join(m for m, v in MUTATIONS.items() if f is None or v[0].endswith(f)))
        return 0
    if len(sys.argv) != 3 or sys.argv[2] not in MUTATIONS:
        print(__doc__)
        return 2
    root, name = sys.argv[1], sys.argv[2]
    fname, desc, edits = MUTATIONS[name]
    path = f"{root}/{fname}"
    src = open(path).read()
    for find, repl, count in edits:
        if src.count(find) != count:
            print(f"mutation {name}: `{find[:60]}` occurs {src.count(find)} times, expected {count}")
            return 1
        src = src.replace(find, repl)
    open(path, "w").write(src)
    print(f"applied {name}: {desc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
