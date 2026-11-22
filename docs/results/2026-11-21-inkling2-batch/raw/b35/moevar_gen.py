"""Variant sources of the shipped NVFP4 expert kernels (kernels/metal/inkling2_moe.metal) for the 1-row ceiling probe
(lane ink-dfuse, measurement only; nothing here is shipped). Each variant is the shipped file with common.h inlined and:
  SG  simdgroups per gate+up threadgroup (shipped 4)      LG  lanes per gate+up row (8)     NB  gate+up chunks ahead (1)
  LD  lanes per down row (4; rows per down threadgroup = 32 / LD)                         NBD down chunks ahead (1)
The shuffle reductions become loops over the lane count (bitwise the same at the shipped 8 / 4). LG / LD other than the
shipped change the chunk order and the reduction tree, so those variants are not bitwise with the shipped kernel.
usage: python3 -I moevar_gen.py KERNEL_DIR OUT_DIR"""
import sys, os
kd, od = sys.argv[1], sys.argv[2]
common = open(os.path.join(kd, "common.h")).read()
moe = open(os.path.join(kd, "inkling2_moe.metal")).read()
def sub(s, a, b, n=1):
    assert s.count(a) == n, (s.count(a), a[:80])
    return s.replace(a, b)
base = sub(moe, '#include "common.h"', common)
# generic shuffle reductions
base = sub(base, """        a += simd_shuffle_xor(a, 1);
        b += simd_shuffle_xor(b, 1);
        a += simd_shuffle_xor(a, 2);
        b += simd_shuffle_xor(b, 2);
        a += simd_shuffle_xor(a, 4);
        b += simd_shuffle_xor(b, 4);""", """        for (ushort m = 1; m < INK_LG; m <<= 1) {
            a += simd_shuffle_xor(a, m);
            b += simd_shuffle_xor(b, m);
        }""")
base = sub(base, """    s += simd_shuffle_xor(s, 1);
    s += simd_shuffle_xor(s, 2);
    return s;""", """    for (ushort m = 1; m < INK_LD; m <<= 1) s += simd_shuffle_xor(s, m);
    return s;""")
# down chunks ahead apart from the gate+up ones
i0 = base.index("inline float ink_dn_row")
i1 = base.index("// down per pair.")
base = base[:i0] + base[i0:i1].replace("INK_NB", "INK_NBD") + base[i1:]
# rows per fused-down threadgroup = 32 / INK_LD
i0 = base.index("inline void ink_dn_f_impl")
i1 = base.index("// y rows -> out")
f = base[i0:i1]
f = sub(f, "const uint n = tg.y * 8u + lane / INK_LD", "const uint n = tg.y * INK_RD + lane / INK_LD")
f = sub(f, "ys[k * 8u + lane / INK_LD]", "ys[k * INK_RD + lane / INK_LD]")
f = sub(f, "if (tid < 8u && tg.y * 8u + tid < H)", "if (tid < INK_RD && tg.y * INK_RD + tid < H)")
f = sub(f, "yv[k] = ys[k * 8u + tid]", "yv[k] = ys[k * INK_RD + tid]")
f = sub(f, "(ulong)t * H + tg.y * 8u + tid", "(ulong)t * H + tg.y * INK_RD + tid")
base = base[:i0] + f + base[i1:]
consts = """constant constexpr uint INK_SG = 4;   // simdgroups per threadgroup of the expert kernels
constant constexpr uint INK_LG = 8;   // lanes per gate+up row
constant constexpr uint INK_LD = 4;   // lanes per down row
constant constexpr uint INK_NB = 1;   // chunks a lane loads ahead per step (3 measured slower: 83 -> 91 us at one NVFP4 row)"""
assert base.count(consts) == 1
DEF = dict(SG=4, LG=8, NB=1, LD=4, NBD=1, XR=4, THR=0, ALU=0, OCC=0, LUT=0)
VARS = {
    "ship": {}, "gu_nb2": dict(NB=2), "gu_lg16": dict(LG=16), "gu_lg16_nb2": dict(LG=16, NB=2), "gu_sg8": dict(SG=8),
    "gu_lg4": dict(LG=4), "dn_nb2": dict(NBD=2), "dn_ld8": dict(LD=8), "dn_ld8_nb2": dict(LD=8, NBD=2),
    "dn_ld16": dict(LD=16), "dn_thr256": dict(THR=256), "dn_ld8_thr256": dict(LD=8, THR=256),
    # N2: one pair per union leader (T = 1 calls: bitwise), down chunks ahead 3 / 4, combinations
    "gu_xr1": dict(XR=1), "gu_xr1_nb2": dict(XR=1, NB=2), "gu_xr1_lg16": dict(XR=1, LG=16), "dn_nb3": dict(NBD=3),
    "dn_nb4": dict(NBD=4), "xr1_dn_nb2": dict(XR=1, NBD=2), "xr1_dn_nb3": dict(XR=1, NBD=3),
    "xr1_dn_ld8_nb2": dict(XR=1, LD=8, NBD=2),
    # lane ink-batch brief 35 diags (NOT bitwise, timing only): ALU = every threadgroup reads the same few weight rows
    # (cache-resident), so the kernel's math, x loads and epilogue run without the DRAM stream
    "gu_alu": dict(ALU=1), "dn_alu": dict(ALU=1), "alu": dict(ALU=1), "xr1_alu": dict(XR=1, ALU=1),
    "xr1_dn_nb3_alu": dict(XR=1, NBD=3, ALU=1),
    # residency probe (timing perturbed): max threadgroups in flight at once, per kernel, via a device atomic at buffer 8
    "occ": dict(OCC=1), "xr1_dn_nb3_occ": dict(XR=1, NBD=3, OCC=1),
    # NVFP4 decode by a 256-entry constant table (byte -> the two e2m1 values x 2^-14, exactly the half-trick values):
    # same values, same fma order, so bitwise; fewer ALU ops per element
    "dn_nb3_thr256": dict(NBD=3, THR=256), "dn_nb2_thr256": dict(NBD=2, THR=256),
    "xr1_dn_nb3_thr256": dict(XR=1, NBD=3, THR=256),
    "lut": dict(LUT=1), "xr1_dn_nb3_lut": dict(XR=1, NBD=3, LUT=1), "xr1_lut_alu": dict(XR=1, LUT=1, ALU=1),
}
xr_line = "constant constexpr uint INK_XR = 4;  // pairs a union leader serves"
assert base.count(xr_line) == 1
for name, o in VARS.items():
    v = dict(DEF, **o)
    c = (f"constant constexpr uint INK_SG = {v['SG']}u;\nconstant constexpr uint INK_LG = {v['LG']}u;\n"
         f"constant constexpr uint INK_LD = {v['LD']}u;\nconstant constexpr uint INK_NB = {v['NB']}u;\n"
         f"constant constexpr uint INK_NBD = {v['NBD']}u;\nconstant constexpr uint INK_RD = 32u / INK_LD;")
    thr = f" THR {v['THR']}" if v["THR"] else ""
    src = base.replace(consts, c).replace(xr_line, f"constant constexpr uint INK_XR = {v['XR']}u;")
    if v["OCC"] and "INK_GU_XR" not in src:
        continue  # the residency probe needs the templated macros (the T1 port)
    if v["OCC"]:
        src = sub(src, "device ushort* h [[buffer(6)]], constant OpP& p [[buffer(7)]],                                      \\\n",
                  "device ushort* h [[buffer(6)]], constant OpP& p [[buffer(7)]], device atomic_uint* occ [[buffer(8)]],\\\n")
        src = sub(src, "        ink_gu_impl<FMT, A16, AX, XR>(x, ids, wts, w13, s13, gsc, h, p, tg, sg, lane);                                  \\\n",
                  "        OCC_IN ink_gu_impl<FMT, A16, AX, XR>(x, ids, wts, w13, s13, gsc, h, p, tg, sg, lane); OCC_OUT \\\n")
        src = sub(src, "device OUT* out [[buffer(6)]], constant OpP& p [[buffer(7)]],                                       \\\n",
                  "device OUT* out [[buffer(6)]], constant OpP& p [[buffer(7)]], device atomic_uint* occ [[buffer(8)]],\\\n")
        src = sub(src, "        ink_dn_f_impl<FMT, A16, OUT, NB>(h, ids, wts, w2, s2, gsc, out, p, ys, tg, sg, lane, tid, nsg);                 \\\n",
                  "        OCC_IN ink_dn_f_impl<FMT, A16, OUT, NB>(h, ids, wts, w2, s2, gsc, out, p, ys, tg, sg, lane, tid, nsg); OCC_OUT \\\n")
        src = sub(src, "#define INK_GU(NAME, FMT, A16, AX) INK_GU_XR", """#define OCC_IN if (sg == 0u && lane == 0u) { const uint c_ = atomic_fetch_add_explicit(&occ[0], 1u, memory_order_relaxed) + 1u; atomic_fetch_max_explicit(&occ[1], c_, memory_order_relaxed); }
#define OCC_OUT threadgroup_barrier(mem_flags::mem_device); if (sg == 0u && lane == 0u) atomic_fetch_sub_explicit(&occ[0], 1u, memory_order_relaxed);
#define INK_GU(NAME, FMT, A16, AX) INK_GU_XR""")
    if v["LUT"]:
        import struct
        def e2m1(c):
            m = c & 7; h = m << 9 | ((c & 8) << 12)
            return struct.unpack("<e", struct.pack("<H", h))[0]
        def lit(x):
            if x == 0: return "-0.0f" if str(x).startswith("-") else "0.0f"
            return repr(float(x)) + "f"
        tab = ", ".join(f"float2({lit(e2m1(b & 15))}, {lit(e2m1(b >> 4))})" for b in range(256))
        src = sub(src, "template <uint FMT>\ninline float ink_chunk_dot(", "constant float2 INK_LUT[256] = {" + tab + "};\n\ntemplate <uint FMT>\ninline float ink_chunk_dot(")
        src = sub(src, """                const uint v = k.w[0][q];
                _Pragma("clang loop unroll(full)")
                for (uint n = 0; n < 4u; ++n) {
                    const uint tt = (v >> (4u * n)) & 0x000F000Fu;  // nibbles n and n + 4
                    const float2 c = float2(as_type<half2>(((tt & 0x00070007u) << 9) | ((tt & 0x00080008u) << 12)));
                    t = fma(c.x, ink_xe(xw, 8u * q + n), t);
                    t = fma(c.y, ink_xe(xw, 8u * q + n + 4u), t);
                }""", """                const uint v = k.w[0][q];
                float ev[8];
                _Pragma("clang loop unroll(full)")
                for (uint j = 0; j < 4u; ++j) {
                    const float2 pr = INK_LUT[(v >> (8u * j)) & 0xFFu];  // DIAG LUT: elements 2j, 2j + 1
                    ev[2u * j] = pr.x;
                    ev[2u * j + 1u] = pr.y;
                }
                _Pragma("clang loop unroll(full)")
                for (uint n = 0; n < 4u; ++n) {
                    t = fma(ev[n], ink_xe(xw, 8u * q + n), t);
                    t = fma(ev[n + 4u], ink_xe(xw, 8u * q + n + 4u), t);
                }""")
    if v["ALU"]:
        gu_on = name != "dn_alu"; dn_on = name != "gu_alu"
        if gu_on: src = sub(src, "const ulong rg = (ulong)e * 2u * I + rr, ru = rg + I;", "const ulong rg = (ulong)(rr % 16u), ru = rg + I;  // DIAG ALU")
        if dn_on: src = sub(src, "const ulong rn = (ulong)id * H + nn;", "const ulong rn = (ulong)(nn % 8u);  // DIAG ALU")
    hdr = f"// variant {name}: SG {v['SG']} LG {v['LG']} NB {v['NB']} LD {v['LD']} NBD {v['NBD']} XR {v['XR']}{thr}{' OCC 1' if v['OCC'] else ''}\n"
    open(os.path.join(od, f"{name}.metal"), "w").write(hdr + src)
    print(name, v)
